import json
import torch
import torch.nn as nn
import joblib
import numpy as np
import math
import pandas as pd
import os
import spacy
import random
import shutil
from lxml import etree
import webbrowser
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

body_width = None
num_nodes = None
num_chars = None

# Load the pretrained spaCy model
nlp = spacy.load("en_core_web_sm")

def verb_ratio(text):
    doc = nlp(text)
    if len(doc) > 5:
        return 0
    
    verb_count = sum(1 for token in doc if token.pos_ == "VERB" and token.lemma_.lower() not in ["username"])
    total_words = sum(1 for token in doc if token.is_alpha)
    
    return verb_count / total_words if total_words > 0 else 0

def is_near_gray(r, g, b, threshold=30, min_val=50, max_val=200):
    return (
        min_val <= r <= max_val and
        min_val <= g <= max_val and
        min_val <= b <= max_val and
        abs(r - g) <= threshold and
        abs(g - b) <= threshold and
        abs(r - b) <= threshold
    )

def color_difference(color1, color2):
    if not all([color1, color2]):
        return 0
    
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    distance = math.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
    max_distance = math.sqrt(3 * 255**2)
    normalized_distance = distance / max_distance
    
    return normalized_distance

class ImprovedTagClassifier(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.4):
        super(ImprovedTagClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.skip1_3 = nn.Linear(512, 128)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.leaky_relu(x1)
        x1 = self.dropout(x1)
        
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.leaky_relu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.fc3(x2)
        skip_x1 = self.skip1_3(x1)
        x3 = x3 + skip_x1
        x3 = self.bn3(x3)
        x3 = self.leaky_relu(x3)
        x3 = self.dropout(x3)
        
        output = self.fc4(x3)
        return output

class MultiLevelTagClassifier:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.preprocessors = {}
        self.label_encoders = {}
        
        self.tag_hierarchy = {
            'DIV': ['DIV', 'FOOTER', 'NAVBAR', 'LIST', 'CARD'],
            'P': ['P', 'LABEL', 'LI', 'TEST'],
            'INPUT': ['INPUT', 'DROPDOWN']
        }
        
        print(f"Using device: {self.device}")
    
    def load_models(self, model_dir='../Models'):
        for parent_tag in self.tag_hierarchy.keys():
            model_path = f'{model_dir}/{parent_tag.lower()}_classifier.pth'
            if os.path.exists(model_path):
                print(f"Loading {parent_tag} model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                model = ImprovedTagClassifier(
                    checkpoint['input_size'], 
                    checkpoint['output_size']
                ).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.models[parent_tag] = model
                self.preprocessors[parent_tag] = checkpoint['preprocessors']
                self.label_encoders[parent_tag] = checkpoint['preprocessors']['label_encoder']
                
                print(f"Loaded {parent_tag} model (Test Accuracy: {checkpoint['test_accuracy']:.4f})")
            else:
                print(f"Model file {model_path} not found!")
    
    def predict_hierarchical(self, sample_data, base_prediction):
        if base_prediction not in self.tag_hierarchy:
            return base_prediction, 1.0
        
        if base_prediction not in self.models:
            print(f"No sub-classifier found for {base_prediction}")
            return base_prediction, 1.0
        
        preprocessors = self.preprocessors[base_prediction]
        sample_df = pd.DataFrame([sample_data])
        
        cat_cols = preprocessors['categorical_cols']
        cont_cols = preprocessors['continuous_cols']
        
        for col in cat_cols + cont_cols:
            if col not in sample_df.columns:
                sample_df[col] = 'unknown' if col in cat_cols else 0
        
        sample_df[cat_cols] = sample_df[cat_cols].astype(str).fillna('unknown')
        X_cat = preprocessors['ohe'].transform(sample_df[cat_cols])
        
        X_cont = preprocessors['imputer'].transform(sample_df[cont_cols])
        X_cont = preprocessors['scaler'].transform(X_cont)
        
        X_processed = np.concatenate([X_cat, X_cont], axis=1)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
        
        model = self.models[base_prediction]
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        predicted_label = preprocessors['label_encoder'].inverse_transform([predicted.cpu().numpy()[0]])[0]
        confidence = probabilities.max().item()
        
        return predicted_label, confidence

def load_model_and_encoders():
    label_encoder = joblib.load("Models/label_encoder.pkl")
    ohe = joblib.load("Models/ohe_encoder.pkl")
    imputer = joblib.load("Models/imputer.pkl")
    scaler = joblib.load("Models/scaler.pkl")

    checkpoint = torch.load("Models/tag_classifier_complete.pth", map_location=torch.device('cpu'))
    model = ImprovedTagClassifier(
        input_size=checkpoint['input_size'],
        output_size=checkpoint['output_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    multi_classifier = MultiLevelTagClassifier()
    multi_classifier.load_models()
    
    return model, label_encoder, ohe, imputer, scaler, multi_classifier

def find_nearest_text_node(node, text_nodes):
    if not text_nodes:
        return 9999999
    
    node_data = node.get("node", {})
    x = node_data.get("x", 0) + node_data.get("width", 0) / 2
    y = node_data.get("y", 0) + node_data.get("height", 0) / 2
    
    min_distance = float('inf')
    for text_node in text_nodes:
        tx, ty = text_node['x'], text_node['y']
        distance = math.sqrt((x - tx)**2 + (y - ty)**2)
        min_distance = min(min_distance, distance)
    
    return min_distance

def extract_features(node, sibling_count=0, prev_sibling_tag=None, parent_height=0, parent_bg_color=None, text_nodes=None):
    global body_width
    global num_nodes
    global num_chars
    
    node_data = node.get("node", {})
    node_type = str(node_data.get("type", ""))
    text = node_data.get("characters", "")
    
    children = node.get("children", [])
    num_direct_children = len(children)
    
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
    
    def count_all_descendants(node):
        count = 0
        for child in node.get("children", []):
            count += 1
            count += count_all_descendants(child)
        return count
    
    def count_chars_to_end(node):
        count = 0
        for child in node.get("children", []):
            node_data = child.get("node", {})
            count += len(node_data.get("characters", ""))
            count += count_chars_to_end(child)
        return count
    
    def get_center_of_weight(node):
        parent_node_data = node.get("node", {})
        parent_x_center = parent_node_data.get("x", 0) + parent_node_data.get("width", 0) / 2
        
        total_area = 0
        total = 0
        for child in node.get("children", []):
            child_node_data = child.get("node", {})
            x = child_node_data.get("x", 0)
            width = child_node_data.get("width", 0)
            height = child_node_data.get("height", 0)
            child_x_center = x + width / 2
            area = width * height
            total += area * child_x_center
            total_area += area
        weighted_x = total / total_area if total_area else parent_x_center
        diff = abs(parent_x_center - weighted_x) / (parent_node_data.get("width", 0) if parent_node_data.get("width", 0) else 1)
        return diff
    
    num_children_to_end = count_all_descendants(node)
    if not num_nodes or num_nodes == 0:
        num_nodes = num_children_to_end
    chars_count_to_end = count_chars_to_end(node)
    if not num_chars or num_chars == 0:
        num_chars = chars_count_to_end
    bg_color = None
    
    feature = {
        "type": node_type,
        "width": node_width/(body_width if body_width else 1),
        "height": node_height/(parent_height if parent_height else node_height if node_height else 1),
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
    feature["nearest_text_node_dist"] = (nearest_text_distance+0.01) / (math.sqrt((node_width+0.001)* (node_height+0.001)) if math.sqrt((node_width+0.001)*(node_height+0.001)) else 1)
    
    return feature

def predict_tag(node, sibling_count, prev_sibling_tag, parent_height, parent_bg_color, text_nodes, model, label_encoder, ohe, imputer, scaler, multi_classifier):
    global body_width
    
    if text_nodes is None:
        def collect_text_nodes(node):
            text_nodes_list = []
            def has_meaningful_text(node_data):
                return node_data.get('type', '') == "TEXT"
            
            node_data = node.get("node", {})
            if has_meaningful_text(node_data):
                text_nodes_list.append({
                    'x': node_data.get("x", 0) + node_data.get("width", 0) / 2,
                    'y': node_data.get("y", 0) + node_data.get("height", 0) / 2
                })
            
            for child in node.get("children", []):
                text_nodes_list.extend(collect_text_nodes(child))
            
            return text_nodes_list
        
        text_nodes = collect_text_nodes(node)
    
    node_data = node.get("node", {})
    figma_type = node_data.get("type", "")
    node_height = node_data.get("height", 0)
    if not body_width or (body_width and body_width == 0):
        body_width = node_data.get("width", 0)
    
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
    
    prev_sib_tag = None
    for i, child in enumerate(node.get("children", [])):
        predict_tag(child, len(node.get("children", []))-1, prev_sib_tag, node_height, bg_color if has_background_color and figma_type != "GROUP" else parent_bg_color, text_nodes, model, label_encoder, ohe, imputer, scaler, multi_classifier)
        prev_sib_tag = child.get("tag", "UNK")

    if figma_type == "GROUP":
        node["tag"] = "DIV"
        node["base_tag"] = "DIV"  # Store base tag
    elif figma_type == "TEXT":
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
    elif node.get("node", {}).get("width", 0) == node.get("node", {}).get("height", 0) and node.get("node", {}).get("width", 0) < 50:
        strokes = node_data.get("strokes", [])
        strokes = [stroke for stroke in strokes if stroke and (color := stroke.get("color")) and color.get("a", 1) > 0]
        fills = node_data.get("fills", [])
        fills = [fill for fill in fills if fill and (color := fill.get("color")) and color.get("a", 1) > 0]
        has_solid_fill = any(fill.get("type") == "SOLID" for fill in fills)
        stroke_color = strokes[0].get("color") if strokes else None
        fill_color = fills[0].get("color") if fills else None
        if not strokes or (stroke_color == fill_color):
            node["tag"] = "LI"
            node["base_tag"] = "LI"
        elif has_solid_fill:
            if node.get("node", {}).get("type", "RECTANGLE") == "RECTANGLE":
                node["tag"] = "CHECKBOX"
                node["base_tag"] = "CHECKBOX"
            elif node.get("node", {}).get("type", "ELLIPSE") == "ELLIPSE":
                node["tag"] = "RADIO"
                node["base_tag"] = "RADIO"
    elif node.get("name", "").startswith("ICON"):
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
    continuous_cols = [col for col in feature.keys() if col not in categorical_cols]
    
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
    
    # Hierarchical prediction in order: P, INPUT, DIV
    predicted_tag = base_predicted_tag
    confidence = base_confidence
    
    for sub_classifier in ['P', 'INPUT', 'DIV']:
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
    
    children = node.get("children", [])
    for i in range(len(children) - 1):
        if (children[i].get("tag") == "P" and 
            children[i+1].get("tag") == "INPUT"):
            children[i]["tag"] = "LABEL"
            children[i]["base_tag"] = "P"  # Ensure base_tag reflects original
    
    for child in children:
        if (child.get("tag") == "DIV" and 
            child.get("node", {}).get("x") == 0.0 and 
            child.get("node", {}).get("y") == 0.0 and 
            abs(child.get("node", {}).get("width", 0) - body_width) < 5
            and child.get("node", {}).get("height", 0) < body_width / 10): 
            child["tag"] = "NAVBAR"
    
    # for child in children:
    #     if (child.get("tag") == "DIV" and 
    #         count_list_items(child) >= 2):
    #         child["tag"] = "UL"
    
    # for child in children:
    #     if child.get("tag") == "DIV":
    #         form_elements = count_form_elements(child)
    #         if form_elements >= 2:
    #             child["tag"] = "FORM"

def count_form_elements(node):
    count = 0
    if not node or "children" not in node:
        return count
    
    for child in node.get("children", []):
        if child.get("tag") == "FORM":
            return 0
        if child.get("tag") in ["INPUT", "BUTTON"]:
            count += 1
    
    for child in node.get("children", []):
        count += count_form_elements(child)
    
    return count

def count_list_items(node):
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

def draw_tags_on_svg_file(data, svg_input_file, svg_output_file=None):
    if svg_output_file is None:
        base, ext = os.path.splitext(svg_input_file)
        svg_output_file = f"{base}_tagged{ext}"
    
    shutil.copy2(svg_input_file, svg_output_file)
    
    parser = etree.XMLParser(remove_blank_text=True)
    tree = etree.parse(svg_output_file, parser)
    root = tree.getroot()
    
    frame_width = root.get('width', str(data.get("node", {}).get("width", "1000"))).replace('px', '')
    frame_height = root.get('height', str(data.get("node", {}).get("height", "1000"))).replace('px', '')
    
    style_element = etree.SubElement(root, 'style')
    style_element.text = """
        .tag-box { stroke: #000000; stroke-width: 1; fill-opacity: 0.3; }
        .tag-text { font-family: Arial; font-size: 10px; }
        .tag-label { fill: white; stroke: #000000; stroke-width: 0.5; rx: 3; ry: 3; }
        .changed-tag { fill: #ff0000; fill-opacity: 0.5; stroke: #ff0000; stroke-width: 2; }
    """
    
    tag_group = etree.SubElement(root, 'g', id="tag-annotations")
    
    tag_colors = {}
    
    def draw_element(element, parent_element):
        if not element or "node" not in element:
            return
            
        tag = element.get("tag", "UNKNOWN")
        base_tag = element.get("base_tag", tag)  # Use tag if base_tag not set
        
        if tag not in tag_colors:
            tag_colors[tag] = generate_random_color()
        color = tag_colors[tag]
        
        node = element["node"]
        x, y = node.get("x", 0), node.get("y", 0)
        width, height = node.get("width", 50), node.get("height", 50)
        
        group = etree.SubElement(parent_element, 'g')
        
        # Highlight if tag changed by sub-model
        is_changed = tag != base_tag
        rect_class = "changed-tag" if is_changed else "tag-box"
        
        rect = etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "class": rect_class,
            "fill": color,
            "stroke": "black" if not is_changed else "#ff0000",
            "stroke-width": "1" if not is_changed else "2"
        })
        
        label_width = max(80, len(tag) * 7 + (len(f"{base_tag} -> ") if is_changed else 0))
        label_height = 40 if not is_changed else 52  # Extra height for changed tags
        
        label_bg = etree.SubElement(group, 'rect', {
            "x": str(x),
            "y": str(y),
            "width": str(label_width),
            "height": str(label_height),
            "rx": "3",
            "ry": "3",
            "fill": "white",
            "fill-opacity": "0.7",
            "stroke": "black",
            "stroke-width": "0.5"
        })
        
        # Show base tag if changed
        if is_changed:
            etree.SubElement(group, 'text', {
                "x": str(x + 3),
                "y": str(y + 12),
                "class": "tag-text",
                "fill": "black"
            }).text = f"{base_tag} -> {tag}"
        else:
            etree.SubElement(group, 'text', {
                "x": str(x + 3),
                "y": str(y + 12),
                "class": "tag-text",
                "fill": "black"
            }).text = tag
        
        etree.SubElement(group, 'text', {
            "x": str(x + 3),
            "y": str(y + 24),
            "class": "tag-text",
            "fill": "black"
        }).text = f"x:{x:.1f}, y:{y:.1f}"
        
        etree.SubElement(group, 'text', {
            "x": str(x + 3),
            "y": str(y + 36),
            "class": "tag-text",
            "fill": "black"
        }).text = f"w:{width:.1f}, h:{height:.1f}"
        
        for child in element.get("children", []):
            if child.get("tag") != "P":
                draw_element(child, group)
    
    draw_element(data, tag_group)
    
    tree.write(svg_output_file, pretty_print=True, encoding='utf-8', xml_declaration=True)
    print(f"SVG visualization created at {svg_output_file}")
    
    webbrowser.open(f"file://{os.path.abspath(svg_output_file)}")

def process_figma_json(input_file, output_file, svg_file=None):
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