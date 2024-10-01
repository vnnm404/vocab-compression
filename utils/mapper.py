import itertools
import json

def generate_attribute_mappings():
    shapes = ['Circle', 'Square', 'Triangle', 'Pentagon', 'Hexagon', 'Octagon', 'Star', 'Cross']
    patterns = ['Solid', 'Striped', 'Dotted', 'Checkered']
    rotations = list(range(0, 360, 60))
    fill_colors = ['Red', 'Green', 'Blue', 'Yellow', 'Purple', 'Orange', 'Pink', 'Brown', 'Cyan', 'Magenta', 'Lime', 'Teal']
    sizes = ['Tiny', 'Small', 'Medium', 'Large', 'Huge']
    textures = ['None', 'Noise']
    opacities = [0.25, 0.5, 0.75, 1.0]
    border_styles = ['Solid', 'Dashed']
    
    attribute_combinations = list(itertools.product(
        shapes, patterns, rotations, fill_colors, sizes, textures, opacities, border_styles
    ))
    
    total_combinations = len(attribute_combinations)
    print(f"Total combinations: {total_combinations}")
    
    idx_to_attributes = {}
    attributes_to_idx = {}
    
    for idx, attributes in enumerate(attribute_combinations):
        idx_to_attributes[idx] = attributes
        attributes_key = '_'.join(map(str, attributes))
        attributes_to_idx[attributes_key] = idx
    
    with open('data/idx_to_attributes.json', 'w') as f:
        json.dump({idx: list(attrs) for idx, attrs in idx_to_attributes.items()}, f)
    
    with open('data/attributes_to_idx.json', 'w') as f:
        json.dump(attributes_to_idx, f)

if __name__ == "__main__":
    pass
