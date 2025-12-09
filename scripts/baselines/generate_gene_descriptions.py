#!/usr/bin/env python3
"""
Generate gene descriptions for Loki model (ST-bank workaround).

Since ST-bank download requires Google Drive permissions, this script creates
gene descriptions from public gene annotation databases for use with Loki.

The descriptions follow the format Loki expects for text encoding.
"""

import argparse
import json
from pathlib import Path

# Gene descriptions based on NCBI Gene and UniProt annotations
# Format: short functional description for CLIP text encoding
GENE_DESCRIPTIONS = {
    # Immunoglobulins
    "IGKC": "immunoglobulin kappa constant region antibody light chain",
    "IGHA1": "immunoglobulin heavy constant alpha 1 antibody secretory",
    "IGHM": "immunoglobulin heavy constant mu antibody B cell membrane",
    "IGHG1": "immunoglobulin heavy constant gamma 1 antibody effector function",
    "JCHAIN": "joining chain of multimeric IgA and IgM immunoglobulin polymerization",

    # Mitochondrial genes
    "MT-ATP6": "mitochondrially encoded ATP synthase membrane subunit 6 oxidative phosphorylation",
    "MT-ND4": "mitochondrially encoded NADH:ubiquinone oxidoreductase core subunit 4",
    "MT-CO3": "mitochondrially encoded cytochrome c oxidase III respiratory chain",
    "MT-CO2": "mitochondrially encoded cytochrome c oxidase II electron transport",
    "MT-CYB": "mitochondrially encoded cytochrome b electron transport chain complex III",
    "MT-ND3": "mitochondrially encoded NADH dehydrogenase 3 complex I",
    "MT-ND4L": "mitochondrially encoded NADH:ubiquinone oxidoreductase core subunit 4L",
    "MT-ND2": "mitochondrially encoded NADH dehydrogenase 2 oxidative phosphorylation",
    "MT-ND1": "mitochondrially encoded NADH dehydrogenase 1 electron transfer",
    "MT-ND5": "mitochondrially encoded NADH:ubiquinone oxidoreductase core subunit 5",

    # Cytoskeleton and structural
    "TMSB4X": "thymosin beta 4 X-linked actin sequestering motility",
    "TMSB10": "thymosin beta 10 actin cytoskeleton organization",
    "ACTB": "actin beta cytoskeleton cell structure motility",
    "VIM": "vimentin intermediate filament mesenchymal cell marker",
    "DES": "desmin intermediate filament muscle cell structural",
    "CFL1": "cofilin 1 actin depolymerizing factor cytoskeleton dynamics",
    "TAGLN": "transgelin actin cross-linking smooth muscle",
    "KRT8": "keratin 8 epithelial intermediate filament cytoskeleton",

    # Extracellular matrix
    "COL3A1": "collagen type III alpha 1 chain extracellular matrix",
    "COL1A1": "collagen type I alpha 1 chain extracellular matrix fibrillar",
    "COL1A2": "collagen type I alpha 2 chain extracellular matrix structural",

    # Iron metabolism
    "FTH1": "ferritin heavy chain 1 iron storage intracellular",
    "FTL": "ferritin light chain iron storage metabolism",

    # Epithelial and cell adhesion
    "CEACAM5": "CEA cell adhesion molecule 5 carcinoembryonic antigen tumor marker",
    "CEACAM6": "CEA cell adhesion molecule 6 epithelial cell adhesion",
    "EPCAM": "epithelial cell adhesion molecule carcinoma marker",
    "PIGR": "polymeric immunoglobulin receptor transcytosis mucosal immunity",
    "CD24": "CD24 molecule cell surface glycoprotein signaling",
    "CD74": "CD74 molecule MHC class II invariant chain antigen presentation",

    # Secreted proteins
    "MUC12": "mucin 12 cell surface associated epithelial",
    "LCN2": "lipocalin 2 neutrophil gelatinase-associated lipocalin inflammation",
    "FCGBP": "Fc fragment of IgG binding protein mucus secretion",
    "LGALS3": "galectin 3 beta-galactoside binding lectin inflammation",
    "OLFM4": "olfactomedin 4 intestinal stem cell marker",
    "CST3": "cystatin C protease inhibitor kidney function",

    # Ribosomal and translation
    "EEF1G": "eukaryotic translation elongation factor 1 gamma protein synthesis",
    "OAZ1": "ornithine decarboxylase antizyme 1 polyamine regulation",

    # Calcium binding
    "S100A6": "S100 calcium binding protein A6 cell cycle progression",
    "S100A10": "S100 calcium binding protein A10 annexin interaction",

    # Signaling and metabolism
    "B2M": "beta-2-microglobulin MHC class I component immune",
    "PYGB": "glycogen phosphorylase B brain form glucose metabolism",
    "PHGR1": "proline, histidine and glycine rich 1",
    "TSPAN8": "tetraspanin 8 cell surface signaling",
    "SYNGR2": "synaptogyrin 2 synaptic vesicle membrane",
    "HSPA8": "heat shock protein family A member 8 chaperone protein folding",
}


def generate_descriptions(gene_list: list, output_path: str = None) -> dict:
    """
    Generate gene descriptions for the given gene list.

    Args:
        gene_list: List of gene symbols
        output_path: Optional path to save descriptions as JSON

    Returns:
        Dictionary mapping gene symbols to descriptions
    """
    descriptions = {}
    missing = []

    for gene in gene_list:
        if gene in GENE_DESCRIPTIONS:
            descriptions[gene] = GENE_DESCRIPTIONS[gene]
        else:
            # Fallback: create generic description
            descriptions[gene] = f"{gene} gene expression protein"
            missing.append(gene)

    if missing:
        print(f"Warning: Using generic descriptions for: {missing}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(descriptions, f, indent=2)
        print(f"Saved {len(descriptions)} gene descriptions to {output_path}")

    return descriptions


def load_gene_list_from_metadata(metadata_path: str) -> list:
    """Load gene list from bin_level gene_names.json or metadata.json"""
    # Try gene_names.json first
    gene_names_path = Path(metadata_path).parent / "gene_names.json"
    if gene_names_path.exists():
        with open(gene_names_path) as f:
            return json.load(f)

    # Fall back to metadata.json
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata.get("gene_names", [])


def main():
    parser = argparse.ArgumentParser(description="Generate gene descriptions for Loki")
    parser.add_argument(
        "--metadata", type=str,
        default="/home/user/img2st-baseline-comparison/data/bin_level/metadata.json",
        help="Path to metadata.json with gene_names"
    )
    parser.add_argument(
        "--output", type=str,
        default="/home/user/img2st-baseline-comparison/data/gene_descriptions.json",
        help="Output path for gene descriptions"
    )
    parser.add_argument(
        "--genes", type=str, nargs="*",
        help="Optional: specify gene list directly"
    )
    args = parser.parse_args()

    if args.genes:
        gene_list = args.genes
    else:
        gene_list = load_gene_list_from_metadata(args.metadata)

    print(f"Generating descriptions for {len(gene_list)} genes...")
    descriptions = generate_descriptions(gene_list, args.output)

    print("\nSample descriptions:")
    for gene in list(descriptions.keys())[:5]:
        print(f"  {gene}: {descriptions[gene]}")


if __name__ == "__main__":
    main()
