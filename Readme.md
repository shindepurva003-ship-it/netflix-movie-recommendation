
# 🎬 Netflix Movie Recommendation System using PySpark ALS

An end-to-end movie recommendation engine built with **Apache Spark (PySpark)** using **Collaborative Filtering** via the **Alternating Least Squares (ALS)** algorithm. The project includes end-to-end data processing, model training, interactive local prototyping with **Gradio**, and export pipelines for a static **Vercel-ready web interface**.

---

## 📌 Table of Contents
1. [Theoretical Background](#-theoretical-background)
   - [What is Collaborative Filtering?](#what-is-collaborative-filtering)
   - [Matrix Factorization & Latent Factors](#matrix-factorization--latent-factors)
   - [Alternating Least Squares (ALS)](#alternating-least-squares-als)
   - [Cold Start Problem & Non-Negative Constraints](#cold-start-problem--non-negative-constraints)
2. [Project Architecture](#-project-architecture)
3. [Implementation Walkthrough](#-implementation-walkthrough)
   - [1. Spark Session Initialization](#1-spark-session-initialization)
   - [2. Data Ingestion & Preprocessing](#2-data-ingestion--preprocessing)
   - [3. ALS Model Configuration & Training](#3-als-model-configuration--training)
   - [4. Generating Recommendations](#4-generating-recommendations)
   - [5. Interactive Prototyping (Gradio)](#5-interactive-prototyping-gradio)
   - [6. Web Deployment Pipeline (Vercel)](#6-web-deployment-pipeline-vercel)
4. [File Structure](#-file-structure)
5. [Setup & Running Locally](#-setup--running-locally)

---

## 🧠 Theoretical Background

### What is Collaborative Filtering?
Recommendation systems generally fall into two categories:
* **Content-Based Filtering:** Recommends items based on explicit feature similarity (e.g., matching genres, actors, directors).
* **Collaborative Filtering:** Makes recommendations based solely on historical interactions (ratings, clicks, views) between users and items. It relies on the assumption that users who agreed in the past will agree in the future.

This project implements **User-Item Collaborative Filtering** with explicit feedback (ratings on a scale of 1 to 5).

### Matrix Factorization & Latent Factors
The interaction history forms a sparse matrix $R$ of dimensions $U \times I$ (where $U$ is the number of users and $I$ is the number of items/movies). Most values are unknown because no user has rated every movie.

Matrix Factorization decomposes this large, sparse matrix into two lower-rank, dense matrices:
* **User Matrix ($P$):** Dimensions $U \times k$
* **Item Matrix ($Q$):** Dimensions $I \times k$

Where $k$ is the number of **latent factors** (hidden patterns such as action intensity, tone, artistic style, pacing). 

The predicted rating $\hat{r}_{u, i}$ of user $u$ for movie $i$ is calculated by the dot product of their latent vectors:

$$\hat{r}_{u, i} = p_u^T q_i$$

### Alternating Least Squares (ALS)
To find $P$ and $Q$, we minimize the regularized squared error loss function across all observed ratings:

$$\min_{P, Q} \sum_{(u, i) \in R_{observed}} (r_{u, i} - p_u^T q_i)^2 + \lambda (\Vert{}p_u\Vert{}_2^2 + \Vert{}q_i\Vert{}_2^2)$$

Where:
* $r_{u, i}$: Ground-truth rating given by user $u$ to movie $i$.
* $\lambda$: Regularization parameter (`regParam`) that prevents overfitting by penalizing large factor weights.

**Why ALS instead of Gradient Descent?**  
Simultaneously optimizing both $P$ and $Q$ is non-convex. **Alternating Least Squares** solves this by:
1. Fixing $Q$ (item matrix) as a constant, turning the objective into a standard quadratic linear regression problem to solve for $P$.
2. Fixing $P$ (user matrix) as a constant and solving for $Q$.
3. Repeating this alternation iteratively until convergence (`maxIter`).

ALS is embarrassingly parallelizable, making it the algorithm of choice for distributed computing frameworks like Apache Spark.

### Cold Start Problem & Non-Negative Constraints
* **Cold Start (`coldStartStrategy="drop"`):** When predicting for users or items not seen during training, standard matrix factorization yields `NaN`. Setting the strategy to `drop` discards these rows to prevent evaluation pipeline failures.
* **Non-Negativity (`nonnegative=True`):** Enforces non-negative constraints on latent factors, ensuring more interpretable factor components and preventing negative predicted scores on positive rating scales.

---

## 🏗️ Project Architecture


```

+-----------------------------------------------------------+
|                   Data Ingestion Layer                   |
|   (30 Movies Metadata + 50 User Ratings via Pandas/Spark) |
+-----------------------------+-----------------------------+
|
v
+-----------------------------------------------------------+
|             PySpark Distributed Pipeline                  |
|    - Schema Inference, Null Checking & Deduplication      |
|    - Explicit Type Casting (int user/item, float rating)  |
+-----------------------------+-----------------------------+
|
v
+-----------------------------------------------------------+
|               Spark MLlib ALS Model                       |
|        - Rank: 10 | MaxIter: 10 | RegParam: 0.1           |
+-----------------------------+-----------------------------+
|
+---------------+---------------+
|                               |
v                               v
+---------------------------+   +---------------------------+
|   Gradio App (Notebook)   |   |   Static Vercel Bundle    |
| - Real-time UI evaluation |   | - Precomputed JSON Matrix |
| - Dynamic user selection  |   | - Netflix UI (HTML/CSS/JS)|
+---------------------------+   +---------------------------+

```

---

## 💻 Implementation Walkthrough

### 1. Spark Session Initialization
We initialize a local Spark context with 4GB driver memory allocated:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Netflix Movie Recommendation System") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

```

### 2. Data Ingestion & Preprocessing

Data consists of two datasets:

* **Movies:** `movieId`, `title`, `genre` (30 curated titles across genres like Action, Comedy Drama, Crime Thriller).
* **Ratings:** `userId`, `movieId`, `rating` (50 interactions from 10 users).

```python
from pyspark.sql.functions import col, count, when

# Ingestion
movies_spark = spark.read.csv("/content/netflix_movies.csv", header=True, inferSchema=True)
ratings_spark = spark.read.csv("/content/netflix_ratings.csv", header=True, inferSchema=True)

# Data Validation: Check for NULLs
ratings_spark.select([count(when(col(c).isNull(), c)).alias(c) for c in ratings_spark.columns]).show()

# Deduplication
movies_spark = movies_spark.dropDuplicates(["movieId"])
ratings_spark = ratings_spark.dropDuplicates(["userId", "movieId"])

# Type Casting for ALS
als_data = ratings_spark.select(
    col("userId").cast("int"),
    col("movieId").cast("int"),
    col("rating").cast("float")
)

```

### 3. ALS Model Configuration & Training

The recommendation engine is configured using Spark MLlib's `ALS` class:

```python
from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    rank=10,                     # Number of latent factors
    maxIter=10,                  # Number of optimization iterations
    regParam=0.1,                # Regularization parameter lambda
    coldStartStrategy="drop",    # Drop records with missing latent factors
    nonnegative=True             # Solve using Non-Negative Matrix Factorization (NMF)
)

model = als.fit(als_data)

```

### 4. Generating Recommendations

We can compute top-$N$ recommendations for a subset of users or batch-compute across all users:

```python
# Generate Top 5 recommendations for User 1
user_recommendations = model.recommendForUserSubset(
    ratings_spark.filter(col("userId") == 1).select("userId"), 
    5
)

# Generate Top 5 recommendations for all users
all_recommendations = model.recommendForAllUsers(5)

```

### 5. Interactive Prototyping (Gradio)

An interactive interface allows instant testing of recommendations directly inside the development environment:

```python
import gradio as gr

def gradio_recommend(user_id):
    try:
        user_id = int(user_id)
        recommendations = recommend_movies(user_id, 5)
        return "\n\n".join(recommendations)
    except:
        return "❌ Please enter a valid User ID from 1 to 10."

with gr.Blocks(title="Netflix Movie Recommendation System") as app:
    gr.Markdown("# 🎬 Netflix Movie Recommendation System\n### Powered by PySpark ALS")
    user_input = gr.Dropdown(choices=list(range(1, 11)), value=1, label="Select User ID")
    recommend_button = gr.Button("🎬 Recommend Movies")
    output = gr.Textbox(label="Recommended Movies", lines=12)
    recommend_button.click(fn=gradio_recommend, inputs=user_input, outputs=output)

app.launch(share=True)

```

### 6. Web Deployment Pipeline (Vercel)

To decouple the heavy Spark runtime from user-facing presentation, the inference matrix is precomputed, mapped to movie titles and genres, and dumped into `recommendations.json`:

```python
all_recommendations = model.recommendForAllUsers(5).collect()
recommendation_data = {}

for row in all_recommendations:
    user_id = row["userId"]
    recommendation_data[str(user_id)] = []
    for rec in row["recommendations"]:
        movie_id = rec["movieId"]
        predicted_rating = float(rec["rating"])
        if movie_id in movie_lookup:
            recommendation_data[str(user_id)].append({
                "movieId": int(movie_id),
                "title": movie_lookup[movie_id]["title"],
                "genre": movie_lookup[movie_id]["genre"],
                "rating": round(predicted_rating, 2)
            })

with open("/content/netflix-vercel/recommendations.json", "w", encoding="utf-8") as f:
    json.dump(recommendation_data, f, indent=2, ensure_ascii=False)

```

The web bundle contains:

* `index.html`: Responsive Netflix-themed dark interface with user selection dropdown.
* `style.css`: Clean Netflix-style black & red layout, movie cards, hover effects, and grid positioning.
* `script.js`: Asynchronous client-side fetch script reading directly from `recommendations.json`.

---

## 📁 File Structure

```
├── netflix_pyspark_notebook.ipynb   # Main Google Colab / Jupyter notebook
├── netflix-vercel/                  # Production export directory
│   ├── index.html                   # Netflix-styled UI
│   ├── style.css                    # Responsive layout styles
│   ├── script.js                    # Dynamic JSON loader
│   └── recommendations.json         # Precomputed ALS recommendation matrices
├── netflix_movies.csv               # Movie metadata (30 movies)
├── netflix_ratings.csv              # User ratings dataset (50 ratings)
└── README.md                        # Documentation

```

---

## 🚀 Setup & Running Locally

### Prerequisites

* Python 3.8+
* Java 8 or 11 (required for Apache Spark runtime)

### Installation

```bash
pip install pyspark gradio pandas numpy

```

### Running the Notebook

Execute all cells sequentially inside Jupyter Notebook or Google Colab:

1. Run dataset creation and Spark session startup.
2. Train the ALS model.
3. Test locally using the embedded Gradio application.
4. Export the `netflix-vercel` bundle.

### Deploying the Front-End to Vercel

1. Download the exported `netflix-movie-recommendation.zip` file.
2. Extract the directory.
3. Deploy to Vercel via the CLI:
```bash
cd netflix-vercel
npx vercel

```
