import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from collections import Counter

URL = "https://cwe.mitre.org/data/definitions/1000.html"
dataset = "AIDev.HUMAN_PULL_REQUEST.parquet"


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text

@dataclass
class CWENode:
    id: str
    name: Optional[str]
    children: List["CWENode"]

def extract_cwe_id_and_name_from_group(group_div):
    a = group_div.select_one("span.Primary a, span.Secondary a")
    if not a:
        return None, None

    raw = a.get_text(" ", strip=True)

    m = re.search(r"\((\d+)\)", raw)
    if not m:
        return None, None

    numeric = m.group(1)
    cwe_id = f"CWE-{numeric}"
    name = raw[: m.start()].strip(" -")

    return cwe_id, name


def parse_group(group_div, pbar: tqdm) -> CWENode:
    pbar.update(1)

    cwe_id, name = extract_cwe_id_and_name_from_group(group_div)

    if not cwe_id:
        html_id = group_div.get("id", "UNKNOWN")
        cwe_id = f"CWE-{html_id}"
        name = name or None

    children: List[CWENode] = []

    for collapseblock in group_div.find_all("div", class_="collapseblock", recursive=False):
        for child_group in collapseblock.find_all("div", class_="group", recursive=False):
            child_node = parse_group(child_group, pbar)
            children.append(child_node)

    return CWENode(id=cwe_id, name=name, children=children)


def build_cwe_1000_tree(html: str) -> CWENode:
    soup = BeautifulSoup(html, "html.parser")

    rel_div = soup.find(id="Relationships")
    if rel_div is None:
        raise RuntimeError("Could not find element with id='Relationships'")

    title_b = rel_div.find("b", string=re.compile(r"\b1000\s*-"))
    if title_b is None:
        raise RuntimeError("Could not find <b> containing '1000 -' in Relationships")

    title_text = title_b.get_text(strip=True)
    m = re.match(r"(\d+)\s*-\s*(.*)", title_text)
    if not m:
        raise RuntimeError(f"Unexpected title format: {title_text!r}")

    root_numeric = m.group(1)
    root_name = m.group(2).strip()
    root_id = f"CWE-{root_numeric}"

    container = title_b.parent
    top_groups = container.find_all("div", class_="group", recursive=False)
    if not top_groups:
        raise RuntimeError("No <div class='group'> found directly under CWE-1000 container")

    all_groups = rel_div.find_all("div", class_="group")
    pbar = tqdm(total=len(all_groups), desc="Parsing CWE-1000 tree")

    children: List[CWENode] = []
    for g in top_groups:
        node = parse_group(g, pbar)
        children.append(node)

    pbar.close()

    root = CWENode(id=root_id, name=root_name, children=children)
    return root


def node_to_dict(node: CWENode) -> dict:
    return {
        "id": node.id,
        "name": node.name,
        "children": [node_to_dict(c) for c in node.children],
    }

def get_nodes_at_depth(root: CWENode, depth: int) -> List[CWENode]:
    if depth < 0:
        return []

    if depth == 0:
        return [root]

    result = []
    frontier = [root]
    current_depth = 0

    while frontier and current_depth < depth:
        next_frontier = []
        for node in frontier:
            next_frontier.extend(node.children)
        frontier = next_frontier
        current_depth += 1

    return frontier


def print_nodes_at_depth(root: CWENode, depth: int):
    nodes = get_nodes_at_depth(root, depth)
    print(f"\nNodes at depth {depth}: (count={len(nodes)})")
    for n in nodes:
        print(f"  {n.id}  —  {n.name}")

def find_node(root: CWENode, target_id: str) -> Optional[CWENode]:
    if root.id == target_id:
        return root
    for child in root.children:
        found = find_node(child, target_id)
        if found:
            return found
    return None

def find_all_first_level_parents(root: CWENode, target_id: str) -> List[CWENode]:
    parents = []
    rst = []

    if root.id == target_id:
        return parents

    for first_level_node in root.children:
        found = find_node(first_level_node, target_id)
        if found:
            parents.append(first_level_node)

    for p in parents:
        rst.append(p.id)
    return rst

def is_ancestor(root: CWENode, ancestor_id: str, descendant_id: str) -> bool:
    ancestor_node = find_node(root, ancestor_id)
    if not ancestor_node:
        return False
    
    found = find_node(ancestor_node, descendant_id)
    return found is not None

def clean_cwe_list(root: CWENode, cwe_list: List[str]) -> List[str]: ## clean ancestors, only keep deepest cwe
    cleaned = set(cwe_list)
    for a in cwe_list:
        for b in cwe_list:
            if a == b:
                continue
            if is_ancestor(root, a, b):
                if a in cleaned:
                    cleaned.remove(a)

    return list(cleaned)

def main():
    counter = Counter()
    missing_counter = 0
    html = fetch_html(URL)
    root = build_cwe_1000_tree(html)
    df = pd.read_parquet(dataset)
    c2 = Counter()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        is_security_patch = row["is_security_patch"]
        cwe_lst = row["cwe_lst"]

        if not is_security_patch:
            continue

        extracted = []
        for d in cwe_lst:
            if not d:
                continue

            cwe_id_raw = str(d.get("cwe_id", "")).strip()
            if not cwe_id_raw.isdigit():
                continue

            extracted.append(f"CWE-{cwe_id_raw}")

        if not extracted:
            missing_counter += 1
            continue

        cleaned = clean_cwe_list(root, extracted)
        c2.update(cleaned)

        if not cleaned:
            continue

        k = len(cleaned)
        weight = 1.0 / k

        for cwe in cleaned:
            parents = find_all_first_level_parents(root, cwe)
            num_parents = len(parents)
            if num_parents > 0:
                per_parent_weight = weight / num_parents
                for p in parents:
                    counter[p] += per_parent_weight
    
    print(sum(counter.values()))

    print()
    print(sum(c2.values()))
    # print(missing_counter)
    df_out = pd.DataFrame([
        {"first_layer_cwe": k, "weight": v}
        for k, v in counter.items()
    ])

    sorted_c2 = c2.most_common()

    for cwe, count in sorted_c2:
        print(f"{cwe}: {count}")


if __name__ == "__main__":
    main()
