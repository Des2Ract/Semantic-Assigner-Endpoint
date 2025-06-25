import json
import torch
import joblib
import numpy as np
import math
import pandas as pd
import os
import sys
import spacy
import random
import shutil
from lxml import etree
import webbrowser

utils_path = os.path.abspath(os.path.join(os.getcwd(), "./Utils/"))
sys.path.append(utils_path)

from utils import verb_ratio, is_near_gray, color_difference,find_nearest_text_node,collect_text_nodes,count_all_descendants,count_chars_to_end,get_center_of_weight
from model_utils import ImprovedTagClassifier, MultiLevelTagClassifier

# Global variables used in normalization
body_width = None
body_height = None
num_nodes = None
num_chars = None

spacy.cli.download("en_core_web_sm")

# Load the pretrained spaCy model
nlp = spacy.load("en_core_web_sm")

def load_model_and_encoders():
    """Load the main model and encoders"""
    label_encoder = joblib.load("./Models/label_encoder.pkl")
    ohe = joblib.load("./Models/ohe_encoder.pkl")
    imputer = joblib.load("./Models/imputer.pkl")
    scaler = joblib.load("./Models/scaler.pkl")

    checkpoint = torch.load("./Models/tag_classifier_complete.pth", map_location=torch.device('cpu'),weights_only=False)
    model = ImprovedTagClassifier(
        input_size=checkpoint['input_size'],
        output_size=checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    multi_classifier = MultiLevelTagClassifier()
    multi_classifier.load_models(model_dir="./Models")
    
    return model, label_encoder, ohe, imputer, scaler, multi_classifier


def extract_features(node, sibling_count=0, prev_sibling_tag=None, parent_height=0, parent_bg_color=None, text_nodes=None):
    global body_width
    global body_height
    global num_nodes
    global num_chars
    
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))
    text = node_data.get("characters", "")
    
    children = node.get("children", [])
    num_direct_children = len(children)
    
    # Initialize child features
    child_1_tag = None
    child_2_tag = None
    child_1_percent = 0
    child_2_percent = 0
    
    node_width = node_data.get("width", 0)
    node_height = node_data.get("height", 0)
    node_area = node_width * node_height
    
    has_placeholder = 0
    is_verb = 0
    if num_direct_children > 0:
        if len(children) >= 1:
            child_1_tag = children[0].get("tag", "")
            child_1_type = children[0].get("node",{}).get("type", "")
            if child_1_type == "TEXT":
                if is_verb == 0:
                    is_verb = verb_ratio(children[0].get("node", {}).get("characters", ""))
                placeholder_fills = children[0].get("node", {}).get("fills", [])
                fills = [fill for fill in placeholder_fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
                for fill in placeholder_fills:
                    if fill.get("type") == "SOLID" and "color" in fill:
                        r, g, b = (
                            int(fill["color"].get("r", 0) * 255),
                            int(fill["color"].get("g", 0) * 255),
                            int(fill["color"].get("b", 0) * 255),
                        )
                        if is_near_gray(r, g, b):
                            has_placeholder = 1
                        break
            child_1_width = children[0].get("node", {}).get("width", 0)
            child_1_height = children[0].get("node", {}).get("height", 0)
            child_1_area = child_1_width * child_1_height
            child_1_percent = (child_1_area / node_area) if node_area > 0 else 0
        
        if len(children) >= 2:
            child_2_tag = children[1].get("tag", "")
            child_2_type = children[1].get("node",{}).get("type", "")
            if child_2_type == "TEXT" and is_verb == 0:
                is_verb = verb_ratio(children[1].get("node", {}).get("characters", ""))
            child_2_width = children[1].get("node", {}).get("width", 0)
            child_2_height = children[1].get("node", {}).get("height", 0)
            child_2_area = child_2_width * child_2_height
            child_2_percent = (child_2_area / node_area) if node_area > 0 else 0
    
    num_children_to_end = count_all_descendants(node)
    if not num_nodes or num_nodes == 0:
        num_nodes = num_children_to_end
    chars_count_to_end = count_chars_to_end(node)
    if not num_chars or num_chars == 0:
        num_chars = chars_count_to_end
    bg_color = None
    
    feature = {
        "type": node_type,
        "y": node_data.get("y", 0) / (body_height if body_height else 1),
        "width": node_width/(body_width if body_width else 1),
        "height": node_height/(parent_height if parent_height else node_height if node_height else 1),
        # "num_direct_children": num_direct_children,
        # "num_children_to_end": num_children_to_end,
        "sibling_count": sibling_count,
        "prev_sibling_html_tag": prev_sibling_tag if prev_sibling_tag else "",
        "has_background_color": 0,
        "border_radius": 0,
        "aspect_ratio": node_width / node_height if node_height > 0 else 0,
        "child_1_html_tag": child_1_tag,
        "child_2_html_tag": child_2_tag,
        "child_1_percentage_of_parent": child_1_percent,
        "child_2_percentage_of_parent": child_2_percent,
        "distinct_background": 0,
        "center_of_weight_diff": get_center_of_weight(node),
        "is_verb": is_verb,
        "has_placeholder": has_placeholder
    }
    
    fills = node_data.get("fills", [])
    fills = [fill for fill in fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
    
    for fill in fills:
        if fill.get("type") == "SOLID" and "color" in fill:
            r, g, b = (
                int(fill["color"].get("r", 0) * 255),
                int(fill["color"].get("g", 0) * 255),
                int(fill["color"].get("b", 0) * 255),
            )
            feature["has_background_color"] = 1
            bg_color = (r, g, b)
            if parent_bg_color:
                bg_difference = color_difference(bg_color, parent_bg_color)
                if bg_difference > 0.25:
                    feature["distinct_background"] = 1   
            break
    
    backgrounds = node_data.get("backgrounds", [])
    for bg in backgrounds:
        if bg.get("type") == "SOLID" and "color" in bg:
            r, g, b = (
                int(bg["color"].get("r", 0) * 255),
                int(bg["color"].get("g", 0) * 255),
                int(bg["color"].get("b", 0) * 255),
            )
            feature["has_background_color"] = 1
            a = min(float(bg["color"].get("a", 1)), float(bg.get("opacity", 1)))
            bg_color = (r*a, g*a, b*a)
            if parent_bg_color:
                bg_difference = color_difference(bg_color, parent_bg_color)
                if bg_difference > 0.2:
                    feature["distinct_background"] = 1  
            break
    
    br_top_left = node_data.get("topLeftRadius", 0)
    br_top_right = node_data.get("topRightRadius", 0)
    br_bottom_left = node_data.get("bottomLeftRadius", 0)
    br_bottom_right = node_data.get("bottomRightRadius", 0)
    
    if any([br_top_left, br_top_right, br_bottom_left, br_bottom_right]):
        feature["border_radius"] = (br_top_left + br_top_right + br_bottom_left + br_bottom_right) / 4
        if feature["border_radius"] >= 50:
            feature["border_radius"] = 0
    
    nearest_text_distance = find_nearest_text_node(node, text_nodes)
    area = node_width * node_height if node_width * node_height > 0 else 0
    feature["nearest_text_node_dist"] = (nearest_text_distance+0.01) / (math.sqrt((area+0.001)) if math.sqrt((area+0.001)) else 1)
    
    return feature

def predict_tag(node, sibling_count, prev_sibling_tag, parent_height, parent_bg_color, text_nodes, model, label_encoder, ohe, imputer, scaler, multi_classifier):
    """Recursively predict HTML tags for a node and its children"""
    global body_width, body_height
    global body_width
    global body_height
    
    # Collect text nodes if not provided
    if text_nodes is None:
        text_nodes = collect_text_nodes(node)
    
    node_data = node.get("node", {})
    figma_type = node_data.get("type", "")
    node_height = node_data.get("height", 0)
    if not body_width or (body_width and body_width == 0):
        body_width = node_data.get("width", 0)
        
    if not body_height or (body_height and body_height == 0):
        body_height = node_data.get("height", 0)
    
    # Extract background color
    fills = node_data.get("fills", [])
    fills = [fill for fill in fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
    has_background_color = False
    bg_color = None
    for fill in fills:
        if fill.get("type") == "SOLID" and "color" in fill:
            r, g, b = (
                int(fill["color"].get("r", 0) * 255),
                int(fill["color"].get("g", 0) * 255),
                int(fill["color"].get("b", 0) * 255),
            )
            has_background_color = True
            bg_color = (r, g, b)                
            break
    backgrounds = node_data.get("backgrounds", [])
    for bg in backgrounds:
        if bg.get("type") == "SOLID" and "color" in bg:
            r, g, b = (
                int(bg["color"].get("r", 0) * 255),
                int(bg["color"].get("g", 0) * 255),
                int(bg["color"].get("b", 0) * 255),
            )
            has_background_color = True
            bg_color = (r, g, b)    
            break
    
    # Recursively process children
    prev_sib_tag = None
    for i, child in enumerate(node.get("children", [])):
        predict_tag(child, len(node.get("children", []))-1, prev_sib_tag, node_height, bg_color if has_background_color and figma_type != "GROUP" else parent_bg_color, text_nodes, model, label_encoder, ohe, imputer, scaler, multi_classifier)
        prev_sib_tag = child.get("tag", "UNK")

    # Assign tags for straightforward cases
    if figma_type == "TEXT":
        node["tag"] = "P"
        node["base_tag"] = "P"
    elif figma_type == "SVG":
        node["tag"] = "SVG"
        node["base_tag"] = "SVG"
    elif figma_type == "VECTOR":
        node["tag"] = "ICON"
        node["base_tag"] = "ICON"
    elif figma_type == "LINE":
        node["tag"] = "HR"
        node["base_tag"] = "HR"
    elif (fills := node_data.get("fills", [])) and any(fill.get("type") == "IMAGE" for fill in fills): 
        node["tag"] = "SVG"
        node["base_tag"] = "SVG"
    elif "icon" in node.get("name", "").lower():
        node["tag"] = "ICON"
        node["base_tag"] = "ICON"
    elif node.get("node", {}).get("width", 0) == node.get("node", {}).get("height", 0) and node.get("node", {}).get("width", 0) < 50:
        strokes = node_data.get("strokes", [])
        strokes = [stroke for stroke in strokes if stroke and (color := stroke.get("color")) and color.get("a", 1) > 0]
        fills = node_data.get("fills", [])
        fills = [fill for fill in fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
        has_solid_fill = any(fill.get("type") == "SOLID" for fill in fills)
        stroke_color = strokes[0].get("color") if strokes else None
        fill_color = fills[0].get("color") if fills else None
        if node.get("node", {}).get("type", "RECTANGLE") == "RECTANGLE" or node.get("node", {}).get("type", "RECTANGLE") == "INSTANCE":
            node["tag"] = "CHECKBOX"
            node["base_tag"] = "CHECKBOX"
        elif node.get("node", {}).get("type", "ELLIPSE") == "ELLIPSE" and stroke_color and (not fill_color or (fill_color and stroke_color != fill_color)):
            node["tag"] = "RADIO"
            node["base_tag"] = "RADIO"
        else:
            node["tag"] = "ICON"
            node["base_tag"] = "ICON"

        
    if node.get("tag", "").upper() != "UNK":
        return
    
    feature = extract_features(
        node,
        sibling_count,
        prev_sibling_tag,
        parent_height,
        parent_bg_color,
        text_nodes
    )
    
    categorical_cols = ['type', 'prev_sibling_html_tag', 'child_1_html_tag', 'child_2_html_tag']
    continuous_cols = [col for col in feature.keys() if col not in categorical_cols and  col not in ['y']]
    
    cat_data = [[feature[col] for col in categorical_cols]]
    X_cat_df = pd.DataFrame(cat_data, columns=categorical_cols)  
    cat_encoded = ohe.transform(X_cat_df)
    
    cont_data = [[feature.get(col, 0) for col in continuous_cols]]
    X_df = pd.DataFrame(cont_data, columns=continuous_cols)
    cont_imputed = imputer.transform(X_df)
    cont_scaled = scaler.transform(cont_imputed)
    
    X_processed = np.concatenate([cat_encoded, cont_scaled], axis=1)
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        base_predicted_tag = label_encoder.inverse_transform(predicted)[0]
        base_confidence = probabilities.max().item()
    
    # Store base predicted tag
    node["base_tag"] = base_predicted_tag
    
    # Add parent_tag_html for sub-classifiers
    feature['parent_tag_html'] = base_predicted_tag
    
    # Hierarchical prediction in order: ICON , P, INPUT, DIV
    predicted_tag = base_predicted_tag
    confidence = base_confidence
    
    for sub_classifier in ['ICON', 'P', 'INPUT', 'DIV']:
        if base_predicted_tag == sub_classifier:
            predicted_tag, confidence = multi_classifier.predict_hierarchical(feature, base_predicted_tag)
            break
    
    X_cat_df["predicted_tag"] = predicted_tag
    X_full_df = pd.concat([X_cat_df, X_df], axis=1)
    X_full_df["predicted_tag"] = predicted_tag
    X_full_df["confidence"] = confidence
    print(f"Predicted tag: {predicted_tag} (Confidence: {confidence:.4f})")
    
    X_full_df.to_csv("features_with_prediction.csv", mode='a', index=False)
    
    node["tag"] = predicted_tag

def post_process_tags(nodes):
    global body_width
    
    if not isinstance(nodes, dict):
        raise ValueError("Expected a dict with 'children' key but got a different structure")
    
    process_nodes(nodes)
    
    return nodes

def process_nodes(node):
    if not node or "children" not in node:
        return
    
    for child in node["children"]:
        process_nodes(child)
    
    # Convert P followed by INPUT to LABEL
    children = node.get("children", [])
    for i in range(len(children) - 1):
        if (children[i].get("tag") == "P" and 
            children[i+1].get("tag") == "INPUT"):
            children[i]["tag"] = "LABEL"
            children[i]["base_tag"] = "P"  # Ensure base_tag reflects original

    # Convert DIV with multiple LI to LIST
    for child in children:
        if (child.get("base_tag") == "DIV" and 
            count_list_items(child) >= 2):
            child["tag"] = "LIST"
        
    # Identify NAVBAR
    for child in children:
        if ((child.get("base_tag") == "DIV" or child.get("tag") == "DIV" or child.get("base_tag") == "LIST" or child.get("tag") == "LIST") and 
            child.get("node", {}).get("y") == 0.0 and 
            child.get("node", {}).get("width", 0) > 0.8 * body_width
            and child.get("node", {}).get("height", 0) < (body_width / 10)):
            child["base_tag"] = "DIV"
            child["tag"] = "NAVBAR"
            
    # Identify FOOTER
    for child in children:
        if (
            (child.get("base_tag") == "DIV" or child.get("tag") == "DIV")
            and child.get("node", {}).get("x") == 0.0
            and abs(child.get("node", {}).get("width", 0) - body_width) < 5
            and child.get("node", {}).get("height", 0) < body_width / 10
            and child.get("node", {}).get("y", 0) > body_width / 2
        ):
            child["base_tag"] = "DIV"
            child["tag"] = "FOOTER"


def count_list_items(node):
    """Count LI elements in direct and first-level indirect children"""
    if not node or "children" not in node:
        return 0
    count = 0
    for child in node.get("children", []):
        if child.get("tag") == "LI":
            count += 1
        elif child.get("tag") != "UL":
            count += sum(1 for grandchild in child.get("children", []) if grandchild.get("tag") == "LI")
    return count

def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgba({r},{g},{b},0.3)"

def parse_length(val):
    try:
        return float(str(val).replace('px', '').replace('%', '').strip())
    except:
        return 1000

def draw_tags_on_svg_file(data, svg_input_file, svg_output_file=None):
    """
    Draw bounding boxes and tags on a copy of an existing SVG file

    Args:
        data : The data containing node information.
        svg_input_file : Path to the original SVG file.
        svg_output_file (optional): Path to save the modified SVG (if None, will use input_file + "_tagged.svg")
    """
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"

    shutil.copy2(svg_input_file, svg_output_file)

    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()

    # Compute max bounds
    def compute_max_bounds(element, max_x=0, max_y=0):
        """Compute maximum bounds for SVG viewport"""
        if "node" in element:
            node = element["node"]
            x = node.get("x", 0)
            y = node.get("y", 0)
            width = node.get("width", 0)
            height = node.get("height", 0)
            max_x = max(max_x, x + width + 100)
            max_y = max(max_y, y + height + 100)
        for child in element.get("children", []):
            max_x, max_y = compute_max_bounds(child, max_x, max_y)
        return max_x, max_y

    max_x, max_y = compute_max_bounds(data)

    # Set SVG bounds and remove overflow styles
    root.attrib["width"] = str(int(max_x))
    root.attrib["height"] = str(int(max_y))
    root.attrib["viewBox"] = f"0 0 {int(max_x)} {int(max_y)}"
    root.attrib.pop("style", None)

    # Remove nested <svg> elements
    for nested_svg in root.findall(".//{http://www.w3.org/2000/svg}svg"):
        parent = nested_svg.getparent()
        if parent is not None:
            parent.remove(nested_svg)

    # Add CSS style block
    style_element = etree.SubElement(root, 'style')
    style_element.text = """
        .tag-box {
            stroke: #000;
            stroke-width: 1;
            fill-opacity: 0.15;
            filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.4));
        }
        .changed-tag {
            fill: #ff0000;
            fill-opacity: 0.25;
            stroke: #ff0000;
            stroke-width: 2;
            filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.4));
        }
        .tag-text {
            font-family: Arial;
            font-size: 12px;
            font-weight: bold;
        }
    """

    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    tag_colors = {}
    drawn_positions = set()

    def draw_element(element, parent_element):
        """Draw a single element with its tag and bounding box"""
        if not element or "node" not in element:
            return

        tag = element.get("tag", "UNKNOWN")
        base_tag = element.get("base_tag", tag)
        color = tag_colors.setdefault(tag, generate_random_color())
        node = element["node"]
        x, y = node.get("x", 0), node.get("y", 0)
        width, height = node.get("width", 50), node.get("height", 50)
        group = etree.SubElement(parent_element, 'g')
        is_changed = tag != base_tag
        rect_class = "changed-tag" if is_changed else "tag-box"

        # Draw main rectangle
        etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": "5",
            "ry": "5",
            "class": rect_class,
            "fill": color,
        })

        # Label text
        lines = [f"{base_tag} → {tag}" if is_changed else tag, f"x:{x:.0f}, y:{y:.0f}"]

        label_padding = 4
        label_line_height = 14
        label_width = max(80, len(lines[0]) * 7)
        label_height = label_line_height * len(lines) + label_padding

        ideal_label_y = y + 2
        label_x = x + 4

        # Avoid overlap
        offset_step = 12
        max_attempts = 10
        for i in range(max_attempts):
            label_y = ideal_label_y - i * offset_step
            key = (round(label_x), round(label_y))
            if key not in drawn_positions:
                drawn_positions.add(key)
                break

        label_y = max(0, label_y)
        label_x = min(label_x, max_x - label_width - 2)

        # Label background
        etree.SubElement(group, 'rect', {
            "x": str(label_x),
            "y": str(label_y),
            "width": str(label_width),
            "height": str(label_height),
            "rx": "3",
            "ry": "3",
            "fill": "#ffffff",
            "fill-opacity": "0.85",
            "stroke": "#000000",
            "stroke-width": "0.5"
        })

        # Label text lines
        for i, line in enumerate(lines):
            etree.SubElement(group, 'text', {
                "x": str(label_x + 5),
                "y": str(label_y + (i + 1) * label_line_height - 4),
                "class": "tag-text",
                "fill": "black"
            }).text = line

    # Priority function
    def is_priority(element):
        """Determine if an element should be drawn in the foreground"""
        tag = element.get("tag", "").lower()
        if tag == "p":
            return None  # Skip <p> entirely
        return tag in {"button", "input", "card", "list", "navbar","footer","checkbox","li"}

    # Recursively draw elements by priority
    def draw_by_priority(element, parent_element, priority: bool):
        """Draw elements by priority to manage layering."""
        if not element or "node" not in element:
            return

        priority_status = is_priority(element)
        if priority_status is None:
            return  # Skip this element and its drawing, but not children

        if priority_status == priority:
            draw_element(element, parent_element)

        for child in element.get("children", []):
            draw_by_priority(child, parent_element, priority)

    # Draw non-priority (background)
    draw_by_priority(data, tag_group, priority=False)

    # Draw priority (foreground)
    draw_by_priority(data, tag_group, priority=True)

    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"✅ SVG visualization created at: {svg_output_file}")
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")

def process_figma_json(input_file, output_file, svg_file=None):
    """
    Process a Figma JSON file, predicting tags for UNK nodes.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        svg_file: Path to the original SVG file (optional)
    """
    model, label_encoder, ohe, imputer, scaler, multi_classifier = load_model_and_encoders()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if os.path.exists('features_with_prediction.csv'):
        os.remove('features_with_prediction.csv')
        
    predict_tag(data, 0, None, None, None, None, model, label_encoder, ohe, imputer, scaler, multi_classifier)
    
    data = post_process_tags(data)
    
    if svg_file:
        svg_output = os.path.splitext(svg_file)[0] + "_tagged.svg"
        draw_tags_on_svg_file(data, svg_file, svg_output)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Processed {input_file}. Output saved to {output_file}")

if __name__ == "__main__":
    process_figma_json("./Data/input4.json", "./Data/output.json", "./Data/input4.svg")