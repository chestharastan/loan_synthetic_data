# Convenience commands for the loan risk pipeline.
# Run `make help` to see everything available.

PYTHON ?= python3

.PHONY: help install data train all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

data:  ## Generate the synthetic dataset
	$(PYTHON) -m data_generation.run

train:  ## Train and evaluate the risk + approval models
	$(PYTHON) -m model_training.train

all: data train  ## Run the full pipeline end to end

clean:  ## Remove generated data, models and metrics
	rm -f data/*.csv models/*.joblib artifacts/*.json artifacts/*.png
