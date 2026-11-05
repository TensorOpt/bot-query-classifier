# bot-query-classifier
Tell apart real search queries from one coming from bot traffic

The classifier decides from the query text alone whether to serve a search query or discard it. The use case is a furniture and home-goods shop. The costly mistake is discarding a real shopper's query, because that can lose a sale, so the false positive rate is capped first and bot detection is maximised within that cap.

> **Status:** this diagram shows the planned design. The project is at an early stage: 3 of 36 tasks are validated, no model has been trained, and there are no results yet. Numbers below are design parameters and targets, not measurements.

## How it works

```mermaid
flowchart TD
    subgraph S1["1. Query logs to labels (separate experiment)"]
        direction TB
        LOGS["Unlabelled query logs<br/>simulated for now: dataset below with labels hidden"]
        EMBED["Character TF-IDF, then SVD to 50 dimensions"]
        HDB["HDBSCAN clustering"]
        SEED["Label a few queries by hand<br/>each cluster takes its seeds' majority label"]
        OPEN["Noise and unseeded clusters<br/>default to legitimate"]
        LOGS --> EMBED --> HDB --> SEED --> OPEN
    end

    subgraph S2["2. Dataset construction"]
        direction TB
        WANDS["WANDS: 480 real customer queries<br/>label 0, legitimate"]
        SYNTH["Generated queries: 480, label 1, illegitimate<br/>120 each: gibberish, injection and probing,<br/>enumeration and scraping, spam"]
        DATA[("Training data: 960 queries")]
        WANDS --> DATA
        SYNTH --> DATA
    end

    subgraph S3["3. Training and evaluation"]
        direction TB
        CV["Group-aware stratified 5-fold CV, repeated 3 times<br/>each fold fits everything below on training data only"]
        TFIDF["Normalise text, character TF-IDF, 1 to 4 characters"]
        SMOTE["SMOTE: measured experiment, off by default"]
        MODEL["Small neural net: one hidden layer of 64 units<br/>compared with logistic regression and a trivial baseline"]
        THRESH["Choose the threshold on inner folds<br/>target: false positive rate of at most 1%"]
        METRICS["Test fold metrics: recall at target FPR, macro F1,<br/>precision, confusion matrix, ROC-AUC"]
        CV --> TFIDF
        TFIDF -.-> SMOTE -.-> MODEL
        TFIDF --> MODEL
        MODEL --> THRESH --> METRICS
    end

    CHECKS["Extra checks, never used for training<br/>50 hand-labelled queries, 100 part numbers,<br/>120 off-topic queries, held-out query families"]
    FINAL["Final model and threshold, fitted on all 960"]

    subgraph S4["4. Classification (illustrative, deployment is out of scope)"]
        direction TB
        QUERY["New search query"]
        SCORE["Normalise, TF-IDF, model score"]
        DECIDE{"Score at or<br/>above threshold?"}
        SERVE["Legitimate: serve"]
        DISCARD["Illegitimate: discard"]
        QUERY --> SCORE --> DECIDE
        DECIDE -- No --> SERVE
        DECIDE -- Yes --> DISCARD
    end

    OPEN -. "possible label source, not built" .-> DATA
    DATA --> CV
    METRICS --> FINAL
    CHECKS --> METRICS
    FINAL --> SCORE
```

Solid lines are the planned pipeline. Dashed lines are optional or not built.
