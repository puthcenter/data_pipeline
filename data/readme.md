# Alibaba Data Flow DAG Dataset

## 1. Dataset Overview
Here we provide the processed datasets. 

These datasets are derived from real-world data flow logs from Alibaba.   The raw data contained cyclic dependencies and isolated vertices, we performed a preprocessing phase to transform the raw flows into DAGs. Due to **commercial confidentiality**, the original topology was unweighted. We assigned synthetic node weights and conflict externalities based on reasonable distributions to simulate realistic industrial scenarios. 

In the provided graph files, all edge costs have been transferred to node costs, please rely solely on node weights/costs for optimization calculations.

## 2. File Naming Convention
The data files follow a strict naming convention that encodes key graph properties:

**Format:** `dag<NodeCount>_<Intensity>_<InstanceID>`

**Example:** `dag121037_MEDIUM_4`

| Segment  | Meaning            | Details                                                      |
| :------- | :----------------- | :----------------------------------------------------------- |
| `dag`    | Prefix             | Indicates a Directed Acyclic Graph file.                     |
| `121037` | Node Count         | The total number of vertices in the graph (e.g., 121,037 nodes). |
| `MEDIUM` | Conflict Intensity | Represents the intensity of negative externalities (conflict strength). |
| `4`      | Instance ID        | The sequence number (e.g., the 4th randomly generated instance). |



