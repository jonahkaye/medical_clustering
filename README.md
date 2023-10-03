###  Medical Clustering Analysis Plan: Uncovering Relationships Between Diagnoses and Underlying Factors

#### Objective:
To explore and identify the relationships between various medical diagnoses and underlying factors by leveraging embeddings and clustering techniques on clinical notes.

#### Phase 1: Data Preparation
- **Data Extraction**:
  - Extract and parse relevant sections (Discharge Diagnosis, Chief Complaint, and History of Present Illness) from the 303 clinical note files.
- **Data Cleaning**:
  - Perform text cleaning (e.g., removing special characters, standardizing terminology) to ensure consistency and quality in the data.

#### Phase 2: Embedding Generation
- **Model Selection**:
  - Choose a pre-trained clinical embedding model (e.g., BioClinicalBERT) suitable for representing medical text.
- **Embedding Creation**:
  - Generate embeddings for each diagnosis and each sentence or relevant text chunk in the HPI sections.
- **Top K**:
  - Choose top k similar embeddings to the diagnosis. 

#### Phase 3: Clustering of Underlying Factors
- **Clustering Algorithm**:
  - Using the top k embeddings for each diagnosis, choose a suitable clustering algorithm (e.g., K-Means, DBSCAN) and determine the optimal number of clusters or parameters.
- **Cluster Analysis**:
  - Generate clusters to understand the common themes or underlying factors represented by each cluster of underlying factors.

#### Phase 4: Associating Diagnoses with Clusters
- **Similarity Computation**:
  - Calculate the similarity between the embeddings of each diagnosis and the centroids of the identified clusters.
- **Association Identification**:
  - Determine which cluster(s) each diagnosis is most similar to, thereby identifying the most related underlying factors.
- **Validation**:
  - Validate the associations with a gut check (GPT-4, doctor, idk?)

#### Phase 5: Communication and Application
- **Visualization**:
  - Develop visualizations (e.g., network graphs, heatmaps) to communicate the relationships and findings effectively.


This analysis plan provides a structured approach to exploring and understanding the relationships between diagnoses and underlying factors in clinical notes, ensuring a balance between data-driven analysis and expert validation.
