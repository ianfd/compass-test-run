# %%
import pandas as pd
import requests
import omnipath as op
import torch


class OmniPathGRNBuilder:
    def __init__(self, gene_list):
        self.genes = gene_list
        self.gene_to_idx = {g: i for i, g in enumerate(gene_list)}

        self.edges = {
        }

    def load_transcriptional_regulation(self):
        print("loading transcriptional regs from omnipdb")

        if "transcriptional" not in self.edges:
            self.edges["transcriptional"] = []

        collectri = op.interactions.CollecTRI()
        interactions = collectri._get(organisms="human", genesymbols=True)

        print(f"retrieved {len(interactions)} tf-target interactions")

        for _, row in interactions.iterrows():
            source = row["source_genesymbol"]
            target = row["target_genesymbol"]

            if source not in self.gene_to_idx or target not in self.gene_to_idx:
                continue

            if "is_stimulation" in row and row["is_stimulation"]:
                sign = 1.0
            elif "is_stimulation" in row and row["is_inhibition"]:
                sign = -1.0
            else:
                sign = 0.0

            confidence = row.get("curation_effort", 1.0)

            self.edges["transcriptional"].append(
                {
                    "source": self.gene_to_idx[source],
                    "target": self.gene_to_idx[target],
                    "source_name": source,
                    "target_name": target,
                    "sign": sign,
                    "confidence": confidence,
                    "directed": True,
                    "databases": row.get("sources", ""),
                }
            )

        print(f"  Added {len(self.edges['transcriptional'])} transcriptional edges")
        print(
            f"    Activation: {sum(1 for e in self.edges['transcriptional'] if e['sign'] > 0)}"
        )
        print(
            f"    Repression: {sum(1 for e in self.edges['transcriptional'] if e['sign'] < 0)}"
        )

        return self

    def load_signaling_network(self):

        if "signaling" not in self.edges:
            self.edges["signaling"] = []

        posttrans = op.interactions.PostTranslational()
        signals = posttrans.get(organisms="human", genesymbols=True)

        for _, row in signals.iterrows():
            source = row["source_genesymbol"]
            target = row["target_genesymbol"]

            if source not in self.gene_to_idx or target not in self.gene_to_idx:
                continue

            if "is_stimulation" in row and row["is_stimulation"]:
                sign = 1.0
            elif "is_stimulation" in row and row["is_inhibition"]:
                sign = -1.0
            else:
                sign = 0.0

            self.edges["signaling"].append(
                {
                    "source": self.gene_to_idx[source],
                    "target": self.gene_to_idx[target],
                    "source_name": source,
                    "target_name": target,
                    "sign": sign,
                    "directed": True,
                    "modification": row.get("modification", "unknown"),
                }
            )

        print(f"  Added {len(self.edges['signaling'])} signaling edges")

        return self

    def load_ppi_network(self, min_confidence=0.4):

        if "ppi" not in self.edges:
            self.edges["ppi"] = []

        all_ints = op.interactions.AllInteractions()
        ppi = all_ints.get(organism="human", genesymbols=True)

        for _, row in ppi.iterrows():
            source = row["source_genesymbol"]
            target = row["target_genesymbol"]

            if source not in self.gene_to_idx or target not in self.gene_to_idx:
                continue

            confidence = row.get("curation_effort", 1.0)
            if confidence < min_confidence:
                continue

            is_directed = row.get("is_directed", False)

            self.edges["ppi"].append(
                {
                    "source": self.gene_to_idx[source],
                    "target": self.gene_to_idx[target],
                    "source_name": source,
                    "target_name": target,
                    "confidence": confidence,
                    "directed": is_directed,
                    "sign": 0.0,
                }
            )

        print(f"  Added {len(self.edges['ppi'])}")

        return self
    
    def load_dorothea_regulons(self, confidence_levels=['A', 'B', 'C']):

        print(f"Loading dorothea TF regulons (conf levels: {confidence_levels})")

        if "dorothea" not in self.edges:
            self.edges["dorothea"] = []

        doroth = op.interactions.Dorothea()
        doro_df = doroth._get(organism="human", genesymbols=True)

        doro_df = doro_df[
            doro_df['source_genesymbol'].isin(self.gene_to_idx) &
            doro_df['target_genesymbol'].isin(self.gene_to_idx)
        ].copy()

        confidence_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}

        for _, row in doro_df.iterrows():
            source = row['source_genesymbol']
            target = row['target_genesymbol']

            confidence_score = row['confidence'].map(confidence_map) / 5.0

            sign = row['sign'].apply(lambda x: 1.0 if x else -1.0)
            
            self.edges["dorothea"].append(
                {
                    "source": self.gene_to_idx[source],
                    "target": self.gene_to_idx[target],
                    "source_name": source,
                    "target_name": target,
                    "confidence": confidence_score,
                    "directed": True,
                    "sign": sign,
                }
            )
        
        print(f"  Added {len(self.edges['dorothea'])} DoRothea edges")
        print(f"    Activation (+1): {sum(1 for e in self.edges['dorothea'] if e['sign'] > 0)}")
        print(f"    Repression (-1): {sum(1 for e in self.edges['dorothea'] if e['sign'] < 0)}")
    
        # Show confidence distribution
        conf_dist = {}
        for e in self.edges['dorothea']:
            level = e['confidence_level']
            conf_dist[level] = conf_dist.get(level, 0) + 1
        print(f"    Confidence distribution: {conf_dist}")

        return self


    def remove_duplicates(self):
        for edge_type in self.edges.keys():
            edge_dict = {}
            for edge in self.edges[edge_type]:
                src, tgt = edge["source"], edge["target"]

                if not edge.get("directed", False):
                    key = tuple(sorted([src, tgt]))
                else:
                    key = (src, tgt)

                if key not in edge_dict or edge.get("confidence", 0) > edge_dict[
                    key
                ].get("confidence", 0):
                    edge_dict[key] = edge

            original_cnt = len(self.edges[edge_type])
            self.edges[edge_type] = list(edge_dict.values())
            removed_cnt = original_cnt - len(self.edges[edge_type])

            if removed_cnt > 0:
                print(f"   {edge_type}: removed {removed_cnt} duplicates")

        return self

    def build_unified_graph(self):
        all_edges = []
        edge_types = []
        edge_attrs = []
        type_mapping = {name: i for i, name in enumerate(self.edges.keys())}

        for edge_type_name, edge_list in self.edges.items():
            type_id = type_mapping[edge_type_name]

            for edge in edge_list:
                src = edge["source"]
                tgt = edge["target"]

                all_edges.append([src, tgt])
                edge_types.append(type_id)

                type_onehot = [0.0] * len(type_mapping)
                type_onehot[type_id] = 1.0

                sign = edge.get("sign", 0.0)
                confidence = edge.get("confidence", 1.0)

                edge_attrs.append(type_onehot + [sign, confidence])
                if not edge.get("directed", False):
                    all_edges.append([tgt, src])
                    edge_types.append(type_id)
                    edge_attrs.append(type_onehot + [sign, confidence])

        edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        print(f"\nFinal unified graph statistics:")
        print(f"   Total edges: {edge_index.shape[1]}")

        for name, type_id in type_mapping.items():
            count = (edge_type == type_id).sum().item()
            pct = 100 * count / edge_index.shape[1]
            print(f"   {name:20s}: {count:6d} ({pct:1f}%)")

        signs = torch.tensor([attr[len(type_mapping)+1] for attr in edge_attr])
        print(f"\nSign distribution:")
        print(f"  Activation (+1):   {(signs == 1.0).sum().item()}")
        print(f"  Repression (-1):   {(signs == -1.0).sum().item()}")
        print(f"  Unknown/PPI (0):   {(signs == 0.0).sum().item()}")

        return edge_index, edge_type, edge_attr

    def get_statistics(self):
        """Print detailed statistics"""
        total_edges = sum(len(edges) for edges in self.edges.values())

        print(f"\n{'=' * 60}")
        print("OmniPath GRN Statistics")
        print(f"{'=' * 60}")
        print(f"Genes in network: {len(self.genes)}")
        print(f"Total edges: {total_edges}")
        print(f"\nEdges by type:")
        for edge_type, edges in self.edges.items():
            print(f"  {edge_type:20s}: {len(edges):6d}")
        print(f"{'=' * 60}\n")
