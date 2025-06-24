import math
import spacy

spacy.cli.download("en_core_web_sm")
# Load the pre-trained spaCy model for English => will bee used in (verb ratio feeature)
nlp = spacy.load("en_core_web_sm")

def verb_ratio(text):
    """Calculate the ratio of verbs in a sentence

    Args:
        text : Sentence

    Returns:
        The ratio of verbs in a sentence => (0 if text length > 5 words or no valid words)
    """
    doc = nlp(text)
    if len(doc) > 5:
        return 0.0
    
    verb_count = sum(1 for token in doc if token.pos_ == "VERB" and token.lemma_.lower() != "username")
    total_words = sum(1 for token in doc if token.is_alpha)  # Count only valid words
    return verb_count / total_words if total_words > 0 else 0.0  # Avoid division by zero

def is_near_gray(r, g, b, threshold = 30, min_val = 50, max_val = 200):
    """Check if an RGB color is near a shade of gray within a specified threshold

    Args:
        threshold : Maximum difference between RGB channels
        min_val : Minimum acceptable channel value
        max_val : Maximum acceptable channel value

    Returns:
        True if the color is near gray (within threshold)

    Notes:
        - Excludes very dark (<50) or very light (>200) gray shades.
    """
    return (
        min_val <= r <= max_val and
        min_val <= g <= max_val and
        min_val <= b <= max_val and
        abs(r - g) <= threshold and
        abs(g - b) <= threshold and
        abs(r - b) <= threshold
    )

def find_nearest_text_node(node, text_nodes):
    """Calculate the Euclidean distance to the nearest text node.

    Args:
        text_nodes : List of text nodes with thier x,y coordinates

    Returns:
        Distance to the nearest text node, or 9999999 if no text nodes exist
    """
    if not text_nodes:
        return 9999999.0  # Default large value if no text nodes
    
    node_data = node.get("node", {})
    x = node_data.get("x", 0) + node_data.get("width", 0) / 2
    y = node_data.get("y", 0) + node_data.get("height", 0) / 2
    
    min_distance = float('inf')
    for text_node in text_nodes:
        tx, ty = text_node['x'], text_node['y']
        distance = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        min_distance = min(min_distance, distance)
    
    return min_distance

def color_difference(color1, color2):
    """
    Calculate a color difference between two RGB colors
    Returns a value between 0 and 1 where 0 means identical and 1 means completely different.
    """
    if not all([color1, color2]):
        return 0
    
    # Extract RGB values
    r1, g1, b1 = color1
    r2, g2, b2 = color2
    
    # Calculate Euclidean distance in RGB space
    distance = math.sqrt((r2-r1)**2 + (g2-g1)**2 + (b2-b1)**2)
    
    # Normalize to 0-1 range
    max_distance = math.sqrt(3 * 255**2)
    normalized_distance = distance / max_distance
    
    return normalized_distance

def collect_text_nodes(node):
    text_nodes_list = []
    # Function to check if a node has meaningful text
    def has_meaningful_text(node_data):
        return node_data.get('type','') == "TEXT"
    
    node_data = node.get("node", {})
    # If this node has meaningful text
    if has_meaningful_text(node_data):
        text_nodes_list.append({
            'x': node_data.get("x", 0) + node_data.get("width", 0) / 2,
            'y': node_data.get("y", 0) + node_data.get("height", 0) / 2,
            'text': node_data.get('characters', '').strip()
        })
    
    # Recursively check children
    for child in node.get("children", []):
        text_nodes_list.extend(collect_text_nodes(child))
    
    return text_nodes_list

def count_all_descendants(node):
    """Count all descendants in the subtree
    """
    count = 0
    for child in node.get("children", []):
        count += 1
        count += count_all_descendants(child)
    return count

def count_chars_to_end(node: dict) -> int:
    """Count total characters in the subtree.

    Returns:
        Total number of characters in the text nodes of the sub tree
    """
    count = 0
    for child in node.get("children", []):
        node_data = child.get("node", {})
        count += len(node_data.get("characters", ""))
        count += count_chars_to_end(child)
    return count

def get_center_of_weight(node):
    """Calculate the center of weight of children in the parent node
    """
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

