import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from flask import Flask, render_template, request

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

label_cols = ["NR-AR","NR-AR-LBD","NR-AhR","NR-Aromatase","NR-ER","NR-ER-LBD","NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE","SR-MMP","SR-p53"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet18_multilabel(num_labels=12):
    m = models.resnet18(pretrained=False)
    nf = m.fc.in_features
    m.fc = nn.Linear(nf,num_labels)
    return m

def load_model(model_path):
    model = get_resnet18_multilabel(num_labels=len(label_cols))
    state_dict = torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class GradCAM:
    def __init__(self,model,target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self,module,input,output):
        self.activations = output.detach()

    def save_gradient(self,module,grad_input,grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self,feature_map,grads):
        alpha = grads.mean(dim=[1,2],keepdim=True)
        weighted = feature_map*alpha
        cam = weighted.sum(dim=0)
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max()!=0:
            cam/=cam.max()
        return cam
    
    def __call__(self,x,target_index):
        output = self.model(x)
        self.model.zero_grad()
        loss = output[0,target_index]
        loss.backward(retain_graph=True)
        fm = self.activations[0]
        gd = self.gradients[0]
        cam = self.generate_cam(fm,gd)
        return cam.cpu().numpy(),output.detach().cpu().numpy()


def overlay_cam_on_image(img_path,cam,alpha=0.5):
    base_img = Image.open(img_path).convert("RGB")
    base_np = np.array(base_img)
    import cv2
    H,W = base_np.shape[:2]
    cam_resized = cv2.resize(cam,(W,H))
    heatmap = plt.cm.jet(cam_resized)[:,:,:3]
    heatmap = (heatmap*255).astype(np.uint8)
    overlay = alpha*heatmap+(1-alpha)*base_np
    overlay = overlay.astype(np.uint8)
    return Image.fromarray(overlay)


def smiles_to_png(smiles,out_path="temp_mol.png",size=(200,200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    AllChem.Compute2DCoords(mol)
    img = Draw.MolToImage(mol,size=size)
    img.save(out_path)
    return out_path

inference_transform = T.Compose([
    T.Resize((200,200)),
    T.ToTensor(),
    T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        smiles = request.form.get("smiles_input","")
        if not smiles:
            return render_template("index.html",error="No SMILES provided!")
        model_choice = request.form.get("model_choice","standard")
        if model_choice=="standard":
            model_path = "models/tox21_multilabel_resnet18.pth"
        else:
            model_path = "models/simclr_tox21_downstream_model.pth"
        try:
            tmp_png = "static/mol.png"
            smiles_to_png(smiles,out_path=tmp_png,size=(200,200))

            if model_path == "models/tox21_multilabel_resnet18.pth":
                loaded_model = load_model(model_path)
            elif model_path == "models/simclr_tox21_downstream_model.pth":
                loaded_model = load_model(model_path)
            
            gradcam = GradCAM(loaded_model,loaded_model.layer4)
            pil_img = Image.open(tmp_png).convert("RGB")
            input_tensor = inference_transform(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = loaded_model(input_tensor)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
            threshold = 0.5
            results = []
            for i,(assay_name,p) in enumerate(zip(label_cols,probs)):
                label_pred = 1 if p>=threshold else 0
                results.append((assay_name,p,label_pred))
            assay_data = []
            for i,(assay_name,p) in enumerate(zip(label_cols,probs)):
                cam,_ = gradcam(input_tensor,target_index=i)
                out_name = f"cam_{i}.png"
                out_path = os.path.join("static",out_name)
                overlay_img = overlay_cam_on_image(tmp_png,cam,alpha=0.5)
                overlay_img.save(out_path)
                assay_data.append({"assay":assay_name,"prob":p,"label":1 if p>=threshold else 0,"heatmap":out_name})
            return render_template("index.html",smiles=smiles,results=results,assay_data=assay_data,model_choice=model_choice)
        except Exception as e:
            return render_template("index.html",error=str(e))
    else:
        return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
