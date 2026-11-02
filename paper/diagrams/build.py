import json
def T(s): return f'<p style="font-size:9pt;text-align:center">{s}</p>'
def box(i,x,y,w,h,text,stroke,fill="#FFFFFF"):
    return {"id":i,"type":"rectangle","boundingBox":{"x":x,"y":y,"w":w,"h":h},"text":T(text),
            "style":{"fill":{"type":"color","color":fill},"stroke":{"color":stroke,"width":2,"style":"solid"},"textColor":"#1F2328"}}
def cont(i,x,y,w,h,title,stroke,fill):
    return {"id":i,"type":"rectangleContainer","boundingBox":{"x":x,"y":y,"w":w,"h":h},
            "containerTitle":{"text":title},
            "style":{"fill":{"type":"color","color":fill},"stroke":{"color":stroke,"width":2,"style":"solid"}}}
def ln(i,a,pa,b,pb,label=None,pos=0.5,kind="straight",dash="solid",color="#3D4450"):
    L={"id":i,"lineType":kind,
       "endpoint1":{"type":"shapeEndpoint","style":"none","shapeId":a,"position":{"x":pa[0],"y":pa[1]}},
       "endpoint2":{"type":"shapeEndpoint","style":"arrow","shapeId":b,"position":{"x":pb[0],"y":pb[1]}},
       "stroke":{"color":color,"width":2,"style":dash}}
    if label: L["text"]=[{"text":f'<span style="font-size:8pt">{label}</span>',"position":pos,"side":"middle"}]
    return L
G,O,B,N="#5B6270","#C8741F","#2F6DB5","#2E8B57"
S=[cont("kA",40,40,1800,300,"Data preparation and automatic staging",G,"#F4F5F7"),
   cont("k1",40,440,520,780,"C1: Conditional GAN augmentation",O,"#FDF1E6"),
   cont("k2",620,440,560,780,"C2: Multi-task RUL and stage prediction",B,"#EAF2FB"),
   cont("k3",1240,440,600,780,"C3: Stage-conditional conformal prediction",N,"#EAF6EE")]
top=["Raw vibration signals (horizontal and vertical), one recording per minute",
     "Feature extraction: 10 time-domain features per channel (d = 20)",
     "Health indicator (RMS): first prediction time and acceleration point",
     "Automatic stage labels (3 stages) and piecewise RUL target",
     "Windows (w = 32); bearing-level train / val / cal / test split"]
for k,t in enumerate(top): S.append(box(f"a{k+1}",120+340*k,120,280,140,t,G))
c1=["Real training windows of all three stages, with stage labels",
    "Generator G(z, stage) and critic D(X, stage) trained with WGAN-GP",
    "Generate near-failure windows; RUL labels drawn from real near-failure RUL",
    "Augmented training set: real windows plus synthetic near-failure windows"]
for k,t in enumerate(c1): S.append(box(f"b{k+1}",120,520+170*k,360,110,t,O))
S+= [box("e1",700,520,400,110,"Shared encoder: two-layer bidirectional LSTM",B),
     box("e2",700,690,400,110,"Temporal attention pooling into a context vector",B),
     box("h1",700,860,170,110,"Stage head: softmax over 3 stages",B),
     box("h2",930,860,170,110,"RUL head: output clipped to [0, 1]",B),
     box("lo",700,1030,400,110,"Joint loss: MSE(RUL) + &lambda; &middot; cross-entropy(stage)",B,"#F6F9FD")]
c3=["Real calibration windows: scores |y &minus; &#375;| grouped by stage",
    "One finite-sample conformal quantile q&#8342; per stage",
    "Stage assigned to each test window",
    "Output: RUL estimate &#375; &plusmn; q&#8342;, coverage &ge; 1 &minus; &alpha; within each stage"]
for k,t in enumerate(c3): S.append(box(f"d{k+1}",1320,520+170*k,440,110,t,N))
Ls=[ln(f"la{k}",f"a{k}",(1,.5),f"a{k+1}",(0,.5)) for k in range(1,5)]
Ls+=[ln(f"lb{k}",f"b{k}",(.5,1),f"b{k+1}",(.5,0)) for k in range(1,4)]
Ls+=[ln(f"ld{k}",f"d{k}",(.5,1),f"d{k+1}",(.5,0)) for k in range(1,4)]
Ls+=[ln("le","e1",(.5,1),"e2",(.5,0)),
     ln("lh1","e2",(.2125,1),"h1",(.5,0)), ln("lh2","e2",(.7875,1),"h2",(.5,0)),
     ln("ll1","h1",(.5,1),"lo",(.2125,0)), ln("ll2","h2",(.5,1),"lo",(.7875,0)),
     ln("xtrain","a5",(.2,1),"b1",(.8,0),"training partition only",0.5,"elbow"),
     ln("xcal","a5",(.8,1),"d1",(.874,0),"real calibration and test bearings",0.45),
     ln("xaug","b4",(1,.5),"e1",(0,.5),None,0.5,"elbow","dashed",O),
     ln("xpred","h2",(1,.5),"d1",(0,.5),None,0.5,"elbow")]
doc={"version":1,"pages":[{"id":"p1","title":"Framework overview","shapes":S,"lines":Ls}]}
for sh in S:
    if sh["id"]!="kA" and not sh["id"].startswith("a"): sh["boundingBox"]["y"]+=120
open("framework.json","w").write(json.dumps(doc))
print(len(S),"shapes",len(Ls),"lines",len(json.dumps(doc)),"bytes")
