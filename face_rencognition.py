import cv2
import face_recognition as fr
font=cv2.FONT_HERSHEY_COMPLEX 


MohFace=fr.load_image_file('C:/Users/NsrO/Documents/Python/demoimages/known/Belbekri Mohammed Bouziane.PNG')
faceLoc=fr.face_locations(MohFace)[0]
MohfaceEnc  = fr.face_encodings(MohFace)[0]


donFace=fr.load_image_file('C:/Users/NsrO/Documents/Python/demoimages/known/Donald Trump.jpg')
faceLoc=fr.face_locations(donFace)[0]
DonfaceEnc  = fr.face_encodings(donFace)[0]


NancyFace=fr.load_image_file('C:/Users/NsrO/Documents/Python/demoimages/known/Nancy Pelosi.jpg')
faceLoc=fr.face_locations(NancyFace)[0]
NancyfaceEnc  = fr.face_encodings(NancyFace)[0] 



KnownEncoding=[DonfaceEnc,NancyfaceEnc,MohfaceEnc]
Names=['Donald Trump','Nancy Pelosi','Belbekri Mohammed Bouziane']


Unknownface=fr.load_image_file('C:/Users/NsrO/Documents/Python/demoimages/unknown/u80.PNG')
UknfaceBGR=cv2.cvtColor(Unknownface,cv2.COLOR_BGR2RGB)
fecaLocations=fr.face_locations(Unknownface)
UknfaceEncs = fr.face_encodings(Unknownface,fecaLocations) 
for fecaLocation,UknfaceEnc in  zip(fecaLocations,UknfaceEncs):
    top,right,buttom,left=fecaLocation
    cv2.rectangle(UknfaceBGR,(left,top),(right,buttom),(100,156,120),5)
    name='Uknown person'
    mat=fr.compare_faces(KnownEncoding,UknfaceEnc)
    if True in mat:
        ma=mat.index(True)
        print(ma)
        print(Names[ma])
        cv2.putText(UknfaceBGR,Names[ma],(left,top-10),font,.5,(100,200,255),1)
    else :
        cv2.putText(UknfaceBGR,name,(left,top-10),font,.5,(100,200,255),1)


    
    
cv2.imshow('Web',UknfaceBGR)
    


cv2.waitKey(15000)
