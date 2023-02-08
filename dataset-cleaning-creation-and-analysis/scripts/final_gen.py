from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = None #very big images, so disable the decompression bomb warning

filename = "final2"

nacimg = Image.open("nac2.png")
tmcimg = Image.open("tmc.png")
ns = 96 #the size of the squares the NAC image must be cut up in
ts = int(ns*tmcimg.width/nacimg.width) #the size of the squares the TMC image must be cut up in

# print(ts) #print TMC width - usually between 20-24

#set up folders
path = os.path.join("images", filename) 
pathh = os.path.join(path, "high")
pathl = os.path.join(path, "low")
os.removedirs(pathh)
os.removedirs(pathl)
os.makedirs(pathh)
os.makedirs(pathl)


# print(nacimg.size)

k = 0
bad = [] #images with too many (> 25%) no-data pixels
for i in range(0, nacimg.width, ns): #divide NAC into strips, width-wise 
	for j in range(0, nacimg.height, ns): #divide NAC into height-wise strips => grid of ns x ns squares
		# print(i, j, k)
		# print((i,j,i+ns, j+ns))
		# print((int(i*ts/ns), int(j*ts/ns), int(i*ts/ns) + ts, int(j*ts/ns) + ts))
		#^ diagnostic print statements
		
		cropped = tmcimg.crop((int(i*ts/ns), int(j*ts/ns), int(i*ts/ns) + ts, int(j*ts/ns) + ts)).resize((24,24)) #crop that specific part of the cropped TMC image and then resize it to 24x24px
		if (cropped.histogram()[0] > 144):
			print(f"bad image for image {k}")
			bad.append(k)
			continue #don't save this image pair
		nacimg.crop((i,j,i+ns, j+ns)).save(f"{pathh}/image_{k}.png") #crop that specific part of the NAC image and save it in the 'high' directory
		cropped.save(f"{pathl}/image_{k}.png") #save the cropped part of the cropped TMC in the 'low' directory
		k+=1
		
		
# print(bad) #print which images had issues



