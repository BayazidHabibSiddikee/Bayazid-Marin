from PIL import Image , ImageDraw
path = r"E:\YT\images\generated-image (3).png"

#Creating an image as im
im = Image.new('RGBA',(200,200),'white')
draw = ImageDraw.Draw(im)

scale = 0.8
threshold = 128
img = Image.open(path).convert("L") #grayscale
img = img.resize((int(img.width*scale),int(img.height*scale)))

for y in range(img.height):
	for x in range(img.width):
		if img.getpixel((x,y))<threshold: #Consider them as black
			draw.point((x,y),fill='black')
im.save('ch17.p2.image.png')
