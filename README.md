# bot-query-classifier
Tell apart real search queries from one coming from bot traffic

The classifier decides from the query text alone whether to serve a search query or discard it. The use case is a furniture and home-goods shop. The costly mistake is discarding a real shopper's query, because that can lose a sale, so the false positive rate is capped first and bot detection is maximised within that cap.

> **Status:** this diagram shows the planned design. The project is at an early stage: data loading, the hand-labelled query set and the query generators are built, the product-name spans are not, no final model has been trained, and there are no final results yet. Numbers below are design parameters and targets, not measurements.

## How it works

```mermaid
flowchart TD
    subgraph S1["1. Query logs to labels (separate experiment)"]
        direction TB
        LOGS["Unlabelled query logs<br/>simulated for now: 480 real and 480 generated queries, labels hidden"]
        EMBED["Character TF-IDF, then SVD to 50 dimensions"]
        HDB["HDBSCAN clustering"]
        SEED["Label a few queries by hand<br/>each cluster takes its seeds' majority label"]
        OPEN["Noise and unseeded clusters<br/>default to legitimate"]
        LOGS --> EMBED --> HDB --> SEED --> OPEN
    end

    subgraph S2["2. Dataset construction"]
        direction TB
        PRODUCTS["WANDS: 42,994 product names"]
        SPANS["Word spans cut from product names: 5,000<br/>label 0, legitimate"]
        SYNTH["Generated queries: 5,000, label 1, illegitimate<br/>gibberish, injection and probing,<br/>enumeration and scraping, spam"]
        DATA[("Training data: 10,000 samples")]
        REAL[("WANDS: 480 real customer queries<br/>label 0, never used for fitting")]
        PRODUCTS --> SPANS --> DATA
        SYNTH --> DATA
    end

    subgraph S3["3. Training and evaluation"]
        direction TB
        CV["Group-aware stratified 5-fold CV, repeated 3 times<br/>each fold splits the generator templates and the real queries"]
        TFIDF["Normalise text, character TF-IDF, 1 to 4 characters"]
        MODEL["Small neural net: one hidden layer of 64 units<br/>compared with logistic regression and a trivial baseline"]
        THRESH["Choose the threshold on 384 real queries<br/>target: false positive rate of at most 1%"]
        METRICS["Test on the other 96 real queries and unseen templates:<br/>recall at target FPR, macro F1,<br/>precision, confusion matrix, ROC-AUC"]
        CV --> TFIDF
        TFIDF --> MODEL
        MODEL --> THRESH --> METRICS
    end

    CHECKS["Per-fold checks<br/>120 off-topic queries, held-out query families,<br/>simulated missing catalogue products,<br/>comparison with real queries as training negatives"]
    FINAL["Final model fitted on spans and generated queries<br/>threshold set on all 480 real queries"]
    FINALCHECKS["Final model checks, never used for training<br/>50 hand-labelled queries, 100 part numbers"]

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
    REAL --> THRESH
    REAL --> METRICS
    METRICS --> FINAL
    CHECKS --> METRICS
    FINAL --> FINALCHECKS
    FINAL --> SCORE
```

Solid lines are the planned pipeline. Dashed lines are optional or not built.
