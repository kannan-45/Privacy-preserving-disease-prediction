"""
Trust-based secure aggregation: each hospital's encrypted model update is
weighted by a trust score derived from its Precision, Recall, F1, and
consistency with its previous round, then combined homomorphically.
"""

from encryption import encrypt_model, decrypt_model


class TrustAggregator:
    """Stateful aggregator that remembers each hospital's metrics across rounds."""

    def __init__(self):
        self.prev_metrics = None

    def aggregate(self, global_model, local_models, sizes, metrics, context):
        shapes = {k: v.shape for k, v in global_model.state_dict().items()}
        enc = [encrypt_model(m, context) for m in local_models]

        weights = []
        for i in range(len(local_models)):
            p = metrics[i]["Precision"]
            r = metrics[i]["Recall"]
            f1 = metrics[i]["F1"]

            balance = 1 - abs(p - r)
            consistency = (
                1 if self.prev_metrics is None
                else 1 - abs(f1 - self.prev_metrics[i]["F1"])
            )

            trust = 0.4 * f1 + 0.3 * balance + 0.3 * consistency
            weights.append(sizes[i] * trust)

        self.prev_metrics = metrics.copy()
        total = sum(weights) + 1e-5

        agg = {}
        for k in shapes:
            num_chunks = len(enc[0][k])
            agg_chunks = []

            for j in range(num_chunks):
                temp = None
                for i in range(len(enc)):
                    val = enc[i][k][j] * (weights[i] / total)
                    temp = val if temp is None else temp + val

                agg_chunks.append(temp)

            agg[k] = agg_chunks

        global_model.load_state_dict(decrypt_model(agg, shapes))
        return global_model
