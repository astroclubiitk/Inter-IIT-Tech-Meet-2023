from PIL import Image
import os
# a = 0
# b = 0
# c = 10
# d = 10
# tifile = Image.open("./image.tif")
filename = "insert nac here"
# tifdim = tifile.size
tifdim = (6500, 162200)
tifgeo = [(285.491278, 29.456955), (286.573891, 2.434635)] #a -> longitude, b -> latitude
imgdim = (5064, 52224)
imggeo = [(286.1, 10.7), (286.17,9.75)] 
imggeo = [(286.0400,27.1500),(286.3800,25.1300)]

scalew = tifdim[0]/(tifgeo[1][0] - tifgeo[0][0])
scaleh = tifdim[1]/(tifgeo[1][1] - tifgeo[0][1])

scale2w = imgdim[0]/(imggeo[1][0] - imggeo[0][0])*0.42
scale2h = imgdim[1]/(imggeo[1][1] - imggeo[0][1])*0.55

endco = ((imggeo[0][0] - tifgeo[0][0])*scalew,
		(imggeo[0][1] - tifgeo[0][1])*scaleh,
		(imggeo[1][0] - tifgeo[0][0])*scalew,
		(imggeo[1][1] - tifgeo[0][1])*scaleh)
print(scalew)
print(scaleh)
print(scale2w)
print(scale2h)
print(endco)
print(endco[2] - endco[0])
print(endco[3] - endco[1])

Image.open("preview.png").crop(tuple([int(x/10) for x in endco])).show()

# tocrop = tifile.crop(enco)
# tocrop.save("test.png")
# tifile = Image.open("./image.tif")
# pixscale = int((1000/nacimg.size[0])*tocrop.size[0])
# 
# nacimg = Image.open(f"./{filename}.IMG.png")
# 
# path = os.path.join("images", filename)
# pathh = os.path.join(path, "high")
# pathl = os.path.join(path, "low")
# os.makedirs(pathh)
# os.makedirs(pathl)
# 
# k = 0
# for i in range(0, nacimg.size[0], 1000):
# 	for j in range(0, nacimg.size[1], 1000):
# 		nacimg.crop((i,j,i+1000, j+1000)).save(f"{pathh}/image{k}")
# 		tocrop.crop((int(i*pixscale/1000),int(j*pixscale/1000),int(i*pixscale/1000)+pixscale,int(j*pixscale/1000)+pixscale)).save(f"{pathl}/image{k}")
# 		k+= 1
