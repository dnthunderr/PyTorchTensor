# Project Instructions

Work for this project will primarily be completed in the Jupyter Notebook titled `project.ipynb`. This notebook contains TODO indicators in four places where you must complete tasks. These four tasks correspond to the four rubric items.

## 1. Choose a Dataset and Explain the Scenario

Consider the dataset choices described on the Project Overview page and select which dataset you are going to use for this project.

In the second cell of the notebook, there is a TODO in a Markdown cell. Double-click on this text to edit it. Write a paragraph explaining which dataset you are using and why.

## 2. Prepare the Dataset for the Custom Query Process

None of the dataset options are in exactly the right format for this process. You need to get them into a format so they can be loaded as a pandas dataframe with a column called `"text"`.

> **Note:** You are not required to use pandas to manipulate the data. You can use Excel/Google Sheets, a text editor, or whatever other software you are comfortable using. If you are using other software to reshape or clean the data, make sure you upload the data file to the data folder then use pandas to load it.

## 3. Perform the Custom Query Process

Integrate your dataset with the custom query code that was provided previously. You can copy and paste as needed. Just make sure that you are using your custom dataset and not one of the datasets used in the course content!

## 4. Write Questions to Demonstrate Custom Performance

In the last cells of the notebook, there are spaces for "Question 1" and "Question 2". Write at least two questions that show how the model answers differently with and without your custom prompt.

---

## Submission Instructions

Once you've met all of the rubric requirements, you can submit your project using one of these options:

- From the workspace using the **SUBMIT PROJECT** button on the workspace page
- As a zip file or a Github repository using the **SUBMIT PROJECT** button on the project submission page

> ⚠️ **Important:** Before you submit your project, remove your OpenAI API key and replace it with `"YOUR API KEY"`. Your reviewer will have their own API key to use when testing your project.

---

## FAQs

### How can I work on this project using my local computer?

A Udacity workspace with all necessary dependencies is available under "Jupyter Workspace - Project". If you would prefer to work on your local computer, you will need:

- Python 3.9
- An environment containing the dependencies listed in `requirements.txt`

Then once your environment is set up, you can download the necessary files directly from the workspace or from the course resources.

> ⚠️ **Important:** This project intentionally uses the legacy version of the openai package, to give you more practice with how these tools work "under the hood". You will encounter issues if you try to complete this using v1 of openai rather than v0.

### I'm stuck on an ImportError, what happened?

The Udacity workspace has openai v0 installed. The approach for importing the library differs between v0 and v1.

**This is the import for v1, which will not work:**

```python
from openai import OpenAI
client = OpenAI()
```

**This is the import for v0, which will work:**

```python
import openai
```

See the [Get a Vocareum OpenAI API Key](https://vocareum.com/docs/getting-started/openai-api-key) page for more details about configuring the package settings for each version.

See the Get a Vocareum OpenAI API Key page for more details about configuring the package settings for each version.

Why do I need to use an outdated version of the openai package? Doesn't that mean my skills will be out of date?
This project is designed to help you understand how a RAG system works "under the hood". Rather than relying on higher-level tools like LangChain or LlamaIndex, we use core Python tools such as pandas and Jupyter Notebooks to walk through each component step by step. This approach prioritizes transparency and learning—once you understand how the pieces fit together, you'll be well-prepared to work with modern frameworks or adapt to new ones as they emerge.

Working with a specific version of a package also reflects common realities in professional environments. In many organizations, software is pinned to a particular version to ensure stability. Learning to work within these constraints is a valuable skill—one that goes beyond simply using the newest API.