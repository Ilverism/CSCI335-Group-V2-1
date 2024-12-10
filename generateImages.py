import os
import io
import random
import math
import requests
from PIL import Image, ImageDraw, ImageFilter
from urllib.parse import urljoin


#Configuration
OUTPUT_DIR = "autogenerated_flag_images"
IMAGE_SIZE = (640, 640)  # (width, height)
NUM_IMAGES = 10


#Possible background types
BACKGROUND_TYPES = ["solid_or_gradient", "realistic", "noise"]

#Sample Country Codes
FLAG_CODES = ["ca", "cn", "de", "in", "jp", "pk", "ru", "kr", "gb", "us"]
FLAG_BASE_URL = "https://flagcdn.com/w320/"  #<-- append country code + ".png"

#Download images from Picsum
REALISTIC_BG_URL = f"https://picsum.photos/{IMAGE_SIZE[0]}/{IMAGE_SIZE[1]}"


#Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def download_image(url):
    
    #Download image from a URL, return a PIL Image

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def generate_solid_or_gradient_background(size):

    #Generate a solid or simple gradient background
    #(randomly choose between solid or gradient)

    #Solid color
    if random.random() < 0.5:
        
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        bg = Image.new("RGB", size, (r, g, b))

    #Simple gradient (with a small number of color steps)
    else:
        
        start_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        end_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        
        #Avoid heavily contrasting colors with a little blending
        mid_color = (
            (start_color[0] + end_color[0]) // 2,
            (start_color[1] + end_color[1]) // 2,
            (start_color[2] + end_color[2]) // 2
        )

        #Create a vertical gradient with up to 3 steps: start_color -> mid_color -> end_color
        bg = Image.new("RGB", size)
        draw = ImageDraw.Draw(bg)
        height = size[1]

        #First half start to mid
        for y in range(height // 2):
            ratio = y / (height / 2)
            r = int(start_color[0] * (1 - ratio) + mid_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + mid_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + mid_color[2] * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))

        #Second half mid to end
        for y in range(height // 2, height):
            ratio = (y - height/2) / (height/2)
            r = int(mid_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(mid_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(mid_color[2] * (1 - ratio) + end_color[2] * ratio)
            draw.line([(0, y), (size[0], y)], fill=(r, g, b))

    return bg


def generate_noise_background(size):
    
    #Generate random white noise background

    width, height = size
    noise_data = bytearray([random.randint(0,255) for _ in range(width*height*3)])
    bg = Image.frombytes('RGB', size, bytes(noise_data))
    return bg


def get_realistic_background(size):
    
    #Download a random realistic background image
    
    img = download_image(REALISTIC_BG_URL)

    #Resize/crop to fit the target size
    img = img.resize(size)

    return img


bg_type_index = 0
def get_random_background(size):

    global bg_type_index

    #Cycle through the background types
    bg_type = BACKGROUND_TYPES[bg_type_index % len(BACKGROUND_TYPES)]
    bg_type_index += 1

    if bg_type == "solid_or_gradient":
        return generate_solid_or_gradient_background(size), bg_type
    elif bg_type == "realistic":
        return get_realistic_background(size), bg_type
    else:
        return generate_noise_background(size), bg_type


def get_random_flag():

    #Download a random flag image

    code = random.choice(FLAG_CODES)
    url = FLAG_BASE_URL + code + ".png"
    flag = download_image(url)
    flag = flag.convert("RGBA")
    return flag, code


def get_flag_from_code(code):
    
    #Download a flag image by country code

    url = FLAG_BASE_URL + code + ".png"
    flag = download_image(url)
    flag = flag.convert("RGBA")
    return flag, code


def place_flag_on_background(bg, flag):
    
    #Place the flag onto the background with random transformations

    bw, bh = bg.size
    fw, fh = flag.size

    #Determine final scale (flag should cover 25-75 percent of the area)
    desired_coverage = random.uniform(0.25, 0.75)
    desired_area = desired_coverage * (bw * bh)
    original_area = (fw * fh)
    scale_factor = math.sqrt(desired_area / original_area)

    new_fw = int(fw * scale_factor)
    new_fh = int(fh * scale_factor)
    flag = flag.resize((new_fw, new_fh))

    #Apply a small rotation
    angle = random.uniform(-20, 20)  # small rotation
    flag = flag.rotate(angle, expand=True, resample=Image.BICUBIC)


    #Apply a a small skew
    skew_factor = random.uniform(-0.1, 0.1)
    width, height = flag.size
    xshift = abs(skew_factor) * height
    transform_matrix = (
        1, skew_factor, -xshift if skew_factor > 0 else 0,
        0, 1, 0
    )
    flag = flag.transform((width, height), Image.AFFINE, transform_matrix, resample=Image.BICUBIC)

    #Random position (ensure flag is at least mostly visible)
    fw2, fh2 = flag.size
    max_x = max(0, bw - fw2)
    max_y = max(0, bh - fh2)

    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    #Composite the flag onto the background
    if (flag.mode != "RGBA"):
        flag = flag.convert("RGBA")

    bg.paste(flag, (pos_x, pos_y), flag.split()[3])  #<-- Use the alpha channel for transparency

    return bg


def main():

    for i in range(NUM_IMAGES):
        for code in FLAG_CODES:

            bg, bg_type = get_random_background(IMAGE_SIZE)
            #flag, flag_code = get_random_flag()
            flag, _ = get_flag_from_code(code)
            composite = place_flag_on_background(bg, flag)
            filename = f"flag_{code}_{bg_type}_{i}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)
            composite.save(filepath, quality=90)
            print(f"Saved {filepath}")


if (__name__ == "__main__"):
    main()