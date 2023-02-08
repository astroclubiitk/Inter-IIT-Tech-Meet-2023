from PIL import Image
import os

filename = "insert nac here"
# tifdim = tifile.size
tifdim = (6500, 162200) #manually entered from XML file
tifgeo = [(285.491278, 29.456955), (286.573891, 2.434635)] #first item of tuple -> longitude, second -> latitude
imgdim = (5064, 52224) #dimensions of NAC - from NAC html page
imggeo = [(286.0400,27.1500),(286.3800,25.1300)] #extents of NAC - from NAC html page

scalew = tifdim[0]/(tifgeo[1][0] - tifgeo[0][0]) #width pixels per longitude degree for TMC
scaleh = tifdim[1]/(tifgeo[1][1] - tifgeo[0][1]) #height pixels per latitude degree for TMC

# scale2w = imgdim[0]/(imggeo[1][0] - imggeo[0][0]) #width pixels per longitude degree for NAC
# scale2h = imgdim[1]/(imggeo[1][1] - imggeo[0][1]) #height pixels per latitude degree for NAC

endco = ((imggeo[0][0] - tifgeo[0][0])*scalew,
		(imggeo[0][1] - tifgeo[0][1])*scaleh,
		(imggeo[1][0] - tifgeo[0][0])*scalew,
		(imggeo[1][1] - tifgeo[0][1])*scaleh)
#coordinates of NAC image inside the TMC image in pixels
# print(scalew)
# print(scaleh)
# print(scale2w)
# print(scale2h)
print(endco)
print(endco[2] - endco[0]) #width of cropped section
print(endco[3] - endco[1]) #height of cropped section

#open the browse file to the given cropped area to make sure everything is okay
Image.open("preview.png").crop(tuple([int(x/10) for x in endco])).show()
