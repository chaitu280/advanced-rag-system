import time
from datetime import datetime


class RAGMonitor:

    def __init__(self):
        self.requests = []

    def start_request(self):
        return time.perf_counter()

    def end_request(
        self,
        start_time,
        question,
        result
    ):
        latency = (
            time.perf_counter() - start_time
        )

        retrieved_results = result.get(
            "retrieved_results",
            []
        )

        claims = result.get(
            "claims",
            []
        )

        validation_status = result.get(
            "validation_status",
            "unknown"
        )

        record = {
            "timestamp": datetime.now().isoformat(),

            "question": question,

            "latency_seconds": round(
                latency,
                4
            ),

            "retrieved_chunks": len(
                retrieved_results
            ),

            "claims": len(
                claims
            ),

            "validation_status":
                validation_status,

            "validation_errors":
                result.get(
                    "validation_errors",
                    []
                )
        }

        self.requests.append(record)

        return record

    def summary(self):

        if not self.requests:
            return {}

        total_requests = len(
            self.requests
        )

        total_latency = sum(
            item["latency_seconds"]
            for item in self.requests
        )

        successful = sum(
            1
            for item in self.requests
            if item["validation_status"]
            == "valid"
        )

        return {
            "total_requests":
                total_requests,

            "average_latency_seconds":
                round(
                    total_latency
                    / total_requests,
                    4
                ),

            "validation_success_rate":
                round(
                    successful
                    / total_requests,
                    4
                )
        }

    def print_summary(self):

        summary = self.summary()

        print("\n")
        print("=" * 60)
        print("RAG MONITORING SUMMARY")
        print("=" * 60)

        print(
            f"Total requests       : "
            f"{summary.get('total_requests', 0)}"
        )

        print(
            f"Average latency      : "
            f"{summary.get('average_latency_seconds', 0)} sec"
        )

        print(
            f"Validation success   : "
            f"{summary.get('validation_success_rate', 0):.2%}"
        )

        print("=" * 60)