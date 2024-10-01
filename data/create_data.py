import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
import random
import os
import colorsys
import multiprocessing as mp

# Global variables (move these outside of functions)
shapes = ['Circle', 'Square', 'Triangle', 'Pentagon', 'Hexagon', 'Octagon', 'Star', 'Cross']
patterns = ['Solid', 'Striped', 'Dotted', 'Checkered']
rotations = list(range(0, 360, 60))
colors = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Pink', 'Brown', 'Cyan', 'Magenta', 'Lime', 'Teal']
sizes = ['Tiny', 'Small', 'Medium', 'Large', 'Huge']
textures = ['None', 'Noise']
opacities = [0.25, 0.5, 0.75, 1.0]
border_styles = ['Solid', 'Dashed']
gradients = ['None']

# Ensure the output directory is global
output_dir = "enhanced_synthetic_dataset"

def create_shape(shape, size, pattern, rotation, edge_color, fill_color, texture, opacity, border_style, text, gradient):
    img = Image.new('RGBA', (100, 100), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    sizes_dict = {'Tiny': 20, 'Small': 35, 'Medium': 50, 'Large': 65, 'Huge': 80}
    radius = sizes_dict[size] // 2
    center = (50, 50)
    
    # Create base shape
    points = get_shape_points(shape, center, radius)
    
    # Apply gradient if specified
    if gradient != 'None':
        img = apply_gradient(img, gradient, fill_color)
        draw = ImageDraw.Draw(img)
    
    # Draw shape
    if border_style == 'Solid':
        if shape == 'Circle':
            draw.ellipse(points, outline=edge_color, fill=fill_color)
        else:
            draw.polygon(points, outline=edge_color, fill=fill_color)
    elif border_style == 'Dashed':
        draw_dashed_polygon(draw, points, edge_color)
        if shape == 'Circle':
            draw.ellipse(points, fill=fill_color)
        else:
            draw.polygon(points, fill=fill_color)
    
    # Apply pattern
    if pattern != 'Solid':
        apply_pattern(draw, pattern, edge_color)
    
    # Apply texture
    if texture != 'None':
        apply_texture(img, texture)
    
    # Add text if specified
    if text:
        add_text(draw, text, center)
    
    # Apply rotation
    img = img.rotate(rotation)
    
    # Apply opacity
    img = img.convert('RGBA')
    data = np.array(img)
    data[..., 3] = data[..., 3] * opacity
    img = Image.fromarray(data)
    
    return img

def get_shape_points(shape, center, radius):
    if shape == 'Circle':
        # Return the bounding box of the circle
        return [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] + radius)
        ]
    elif shape == 'Square':
        return [
            (center[0] - radius, center[1] - radius),
            (center[0] + radius, center[1] - radius),
            (center[0] + radius, center[1] + radius),
            (center[0] - radius, center[1] + radius)
        ]
    elif shape == 'Triangle':
        return [
            (center[0], center[1] - radius),
            (center[0] - radius, center[1] + radius),
            (center[0] + radius, center[1] + radius)
        ]
    elif shape in ['Pentagon', 'Hexagon', 'Octagon']:
        n_sides = {'Pentagon': 5, 'Hexagon': 6, 'Octagon': 8}[shape]
        angle = 2 * np.pi / n_sides
        return [
            (
                center[0] + radius * np.cos(i * angle - np.pi / 2),
                center[1] + radius * np.sin(i * angle - np.pi / 2)
            )
            for i in range(n_sides)
        ]
    elif shape == 'Star':
        points = []
        for i in range(10):
            angle = i * np.pi * 2 / 10 - np.pi / 2
            r = radius if i % 2 == 0 else radius / 2
            points.append((
                center[0] + r * np.cos(angle),
                center[1] + r * np.sin(angle)
            ))
        return points
    elif shape == 'Cross':
        third = radius / 3
        return [
            (center[0] - third, center[1] - radius),
            (center[0] + third, center[1] - radius),
            (center[0] + third, center[1] - third),
            (center[0] + radius, center[1] - third),
            (center[0] + radius, center[1] + third),
            (center[0] + third, center[1] + third),
            (center[0] + third, center[1] + radius),
            (center[0] - third, center[1] + radius),
            (center[0] - third, center[1] + third),
            (center[0] - radius, center[1] + third),
            (center[0] - radius, center[1] - third),
            (center[0] - third, center[1] - third)
        ]

def draw_dashed_polygon(draw, points, color):
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        draw_dashed_line(draw, start, end, color)

def draw_dashed_line(draw, start, end, color, dash_length=5):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    dashes = max(int(length / dash_length), 1)
    for i in range(dashes):
        s = i / dashes
        e = (i + 0.5) / dashes
        sx = x1 + s * dx
        sy = y1 + s * dy
        ex = x1 + e * dx
        ey = y1 + e * dy
        draw.line([(sx, sy), (ex, ey)], fill=color)

def apply_pattern(draw, pattern, color):
    if pattern == 'Striped':
        for i in range(0, 100, 4):
            draw.line([(i, 0), (i, 100)], fill=color, width=2)
    elif pattern == 'Dotted':
        for i in range(0, 100, 10):
            for j in range(0, 100, 10):
                draw.ellipse([i-2, j-2, i+2, j+2], fill=color)
    elif pattern == 'Checkered':
        for i in range(0, 100, 20):
            for j in range(0, 100, 20):
                draw.rectangle([i, j, i+10, j+10], fill=color)

def apply_texture(img, texture):
    if texture == 'Noise':
        data = np.array(img)
        noise = np.random.rand(*data.shape[:2]) * 50
        data[:, :, :3] = np.clip(data[:, :, :3] + noise[:, :, np.newaxis], 0, 255)
        return Image.fromarray(data.astype('uint8'), 'RGBA')
    return img

def add_text(draw, text, center):
    font = ImageFont.load_default()
    # Use textbbox if available, otherwise use textsize
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left
        text_height = bottom - top
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)
    position = (center[0] - text_width // 2, center[1] - text_height // 2)
    draw.text(position, text, fill="black", font=font)

def apply_gradient(img, gradient_type, base_color):
    width, height = img.size
    base_rgb = ImageColor.getrgb(base_color)
    base_hsv = colorsys.rgb_to_hsv(*[x / 255.0 for x in base_rgb])
    
    if gradient_type == 'Radial':
        for y in range(height):
            for x in range(width):
                distance = ((x - width / 2)**2 + (y - height / 2)**2)**0.5
                factor = distance / (width / 2)
                new_v = max(0, min(1, base_hsv[2] * (1 - factor)))
                rgb = colorsys.hsv_to_rgb(base_hsv[0], base_hsv[1], new_v)
                img.putpixel((x, y), tuple(int(c * 255) for c in rgb))
    elif gradient_type == 'Linear':
        for y in range(height):
            factor = y / height
            new_v = max(0, min(1, base_hsv[2] * (1 - factor)))
            rgb = colorsys.hsv_to_rgb(base_hsv[0], base_hsv[1], new_v)
            for x in range(width):
                img.putpixel((x, y), tuple(int(c * 255) for c in rgb))
    
    return img

def generate_image(i):
    try:
        # Ensure different random seed for each process
        random.seed(os.getpid() + i)

        shape = random.choice(shapes)
        pattern = random.choice(patterns)
        rotation = random.choice(rotations)
        edge_color = random.choice(colors)  # Random edge color, but will not be included in labels
        fill_color = random.choice(colors)
        size = random.choice(sizes)
        texture = random.choice(textures)
        opacity = random.choice(opacities)
        border_style = random.choice(border_styles)
        gradient = random.choice(gradients)

        text = ''
        if random.random() < 0.3:  # 30% chance of having text
            text = random.choice(['A', 'B', 'C', '1', '2', '3'])

        img = create_shape(
            shape, size, pattern, rotation,
            edge_color, fill_color, texture, opacity, border_style, text, gradient
        )

        # Filename excludes edge_color
        filename = f"{i:06d}_{shape}_{pattern}_{rotation}_{fill_color}_{size}_{texture}_{opacity}_{border_style}.png"
        img.save(os.path.join(output_dir, filename), "PNG")

        # Progress update
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} images")

    except Exception as e:
        print(f"Error generating image {i}: {e}")

def generate_dataset(num_images, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Use multiprocessing Pool to parallelize
    num_processes = mp.cpu_count()
    print(f"Using {num_processes} processes for image generation.")

    with mp.Pool(processes=num_processes) as pool:
        pool.map(generate_image, range(num_images))

    print("Enhanced dataset generation complete!")

if __name__ == "__main__":
    num_images = 10  # Adjust as needed
    output_dir = "enhanced_synthetic_dataset"
    generate_dataset(num_images, output_dir)
