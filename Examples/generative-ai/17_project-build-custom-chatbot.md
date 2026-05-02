# Project: Build Your Own Custom Chatbot

In this project you will change the dataset to build your own custom chatbot. You will build the chatbot manually using just basic packages like `openai` and `pandas`, not using frameworks like LangChain, in order to gain a deeper understanding of how these systems work "under the hood".

## What Will You Build?

When you have completed this project, you will have a custom OpenAI chatbot using a scenario of your choice. You will be responsible for:

- Selecting a data source
- Explaining why the data source is appropriate for the task
- Incorporating the data source into the custom chatbot code
- Writing questions to demonstrate the performance of the custom prompt

## Data Sources

There are two main data sources we recommend using for this project: Wikipedia API and CSV data.

### Wikipedia API

The Wikipedia API will be most similar to the examples shown in the demos and exercises you have previously seen. You can use any article other than:

- 2022
- 2023 Syria-Turkey Earthquake

as long as it fulfills the requirements.

### CSV Data

We have provided a data directory containing CSV files for this project:

- **2023_fashion_trends.csv** - Reports and quotes about fashion trends for 2023. Each row includes the source URL, article title, and text snippet.
- **character_descriptions.csv** - Character descriptions from theater, television, and film productions. Each row contains the name, description, medium, and setting. All characters were invented by an OpenAI model.
- **nyc_food_scrap_drop_off_sites.csv** - Locations, hours, and other information about food scrap drop-off sites in New York City. This information was retrieved in early 2023, and you can also get the latest version from the open data portal.

### Custom Data Sources

You may also source your own data. For example:

- Web scraping
- Local documents or databases

**Requirements:** At least 20 rows of text data

**Note:** OpenAI language models are not optimized for numeric or logical reasoning, so number-heavy data like budgets, sensor data, or inventory are not appropriate.

## Custom Scenario

In addition to the technical component of preparing and incorporating a new dataset, you need to explain why this dataset is appropriate for the task. If the model responds the same way regardless of whether custom data is provided, that means the dataset was not appropriate for the task.

You will explain your dataset choice in two ways:

1. **At the start of the notebook:** Write a short paragraph describing your dataset choice and setting up the scenario of when this customization would be useful.

2. **At the end of the notebook:** Demonstrate the model Q&A before and after the customization has been performed in order to highlight the changes.