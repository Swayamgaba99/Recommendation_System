# 🚀 Neo4j to TensorFlow GNN Pipeline

A robust pipeline that extracts data from a **Neo4j graph database** and constructs a **TensorFlow GraphTensor** using **TensorFlow GNN**. This project enables the development of Graph Neural Networks for user behavior modeling, recommendation systems, and graph-structured data learning.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Graph Schema](#graph-schema)
- [Features](#features)
- [Dependencies](#dependencies)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Example Applications](#example-applications)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## 🧠 Overview

This project demonstrates:

- Extracting **nodes** and **edges** from Neo4j using Cypher queries.
- Structuring data as `GraphTensor` using TensorFlow GNN.
- Preparing data for machine learning tasks such as **node classification**, **link prediction**, and **graph-based recommendation**.

---

## 🕸️ Graph Schema

### 🧩 Node Types

| Node Type  | Attributes                            |
|------------|----------------------------------------|
| `User`     | `userId`, `username`, `email`          |
| `Stream`   | `streamId`, `streamName`, `views`, `likes`, `comments` |
| `Category` | `name`                                 |
| `Location` | `city`                                 |

### 🔗 Edge Types

| Relationship        | Source → Target     |
|---------------------|---------------------|
| `FOLLOWS`           | User → User         |
| `VIEWED`            | User → Stream       |
| `ENGAGES_WITH`      | User → Stream       |
| `BELONGS_TO`        | Stream → Category   |
| `LIVES_IN`          | User → Location     |

---

## ✨ Features

- Full Neo4j to TensorFlow GNN integration
- Supports heterogeneous graphs (multiple node/edge types)
- Ready-to-train `GraphTensor` object creation
- Easily extendable for downstream GNN models

---

## 🧱 Dependencies

Install the required Python packages:

```bash
pip install neo4j pandas tensorflow tensorflow_gnn

