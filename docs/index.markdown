---
layout: post
title:  "Autonomous Systems Evaluation Standard"
date:   2024-10-31 11:35:57 +0000
excerpt: This document outlines the minimal requirements and recommendations for the submission of a new evaluation to the UK AI Safety Institute's (AISI) Autonomous Systems evaluation suite.
authors: "Sid Black, Jacob Arbeid, Jay Bailey, Alan Cooney, Oliver Sourbut, Michael Schmatz, Jasmine Wang"

---

<small class="small post-meta">{{ page.authors }}</small>

---

* This will become a table of contents (this text will be scrapped).
{:toc}

## Introduction

This document outlines the minimal requirements and recommendations for the submission of a new evaluation to the [UK AI Safety Institute's](https://www.aisi.gov.uk/) (AISI) Autonomous Systems evaluation suite. The aim of this standard is to maintain a high level of quality across our entire evaluation suite.

## Submission Requirements

### Required Standards

The criteria in this section must be included for your submission to be accepted.

#### Inspect

All evaluations must be built using [Inspect](https://inspect.ai-safety-institute.org.uk/) – an open-source framework for frontier model evaluations, created by the AI Safety Institute (AISI).

If you’re not familiar with Inspect yet, please do read through the [Inspect documentation](https://inspect.ai-safety-institute.org.uk/) first and then come back to this document.

#### Code and Repository Structure

All submissions must match the structure of the cookiecutter template in the as-evaluation-standard GitHub repository at <https://github.com/UKGovernmentBEIS/as-evaluation-standard>.  

To use this, head over to the repository and follow the instructions in the [Evaluation Template](https://github.com/UKGovernmentBEIS/as-evaluation-standard?tab=readme-ov-file#evaluation-template) section.

Any assets (files that need to be copied over to the evaluation [sandbox](https://inspect.ai-safety-institute.org.uk/agents.html#sec-sandbox-environments)) should be included in the assets folder.

Your code should pass all the automated checks in the evaluation standard repository. You can verify this using [the built-in GitHub Action](https://github.com/UKGovernmentBEIS/as-evaluation-standard/blob/main/.github/workflows/ci.yml), or alternatively by running `poe check` from the template repository on your codebase. These checks include type checks (we require all our evaluations be type annotated), ruff formatting checks, and others.

We require that a few specific files are included, which we detail below:

* **README.md:** A `README.md` file should be included, with all the details filled in as per our template README file. We ask that developers include our unaltered [canary string](https://github.com/UKGovernmentBEIS/as-evaluation-standard/blob/f033cbca34a49f78624269c92f059630794d2999/example_evaluations/simple/README.md?plain=1#L5) - `c65e97c1-4e5b-401d-82f2-4581ce8f14d7` - somewhere in the README. This is required to detect if the repository has been included in training corpora.
* **CONTRIBUTING.md:** Once the evaluation is submitted to us, we take on the maintenance workload, and we appreciate any help we can get with this! Consequently, we require that developers provide a `CONTRIBUTING.md` file, which should contain information that will simplify maintenance and future development of the evaluation. Specifically, it should contain all the fields detailed in our template CONTRIBUTING file.
* **METADATA.json:** Structured evaluation metadata that conforms to the data model defined in METADATA.json in the task-standard GitHub repo.

#### Scoring

Scoring for your evaluation should be fully automated, requiring no manual grading steps.

#### Quality Assurance

You must provide evidence that you have performed quality assurance (QA) checks on your evaluation. This means evidence that either a frontier model or a human can complete at least a single sample of the evaluation. This should be evidenced by including an Inspect log file/files at `<evaluation_directory>/QA/<log_file_name>.json`.  

If the evaluation has distinct categories, you should provide a QA sample for each category.

In addition to this, you should run your evaluation on a frontier model and manually check the logs to ensure there are no model API errors, tool issues or other failures that may result in models underperforming on your tasks.  

#### Custom Tests

Each evaluation must include evaluation-specific tests. The contents of these tests may vary depending on the evaluation, but at a minimum, we ask that:

* Any custom tools or scorers implemented for the evaluation have unit tests.
* For evaluations that require dynamically accessing external resources (for example a [huggingface dataset](https://inspect.ai-safety-institute.org.uk/datasets.html#sec-hugging-face-datasets)), there should be tests that verify this resource acquisition and the task setup are working correctly.

#### Ethical and Safety Guidelines

You should not encourage the agent to do anything unsafe, illegal, or unethical in the real world. If you need to test for potentially unsafe capabilities, consider using mocking/simulation, or testing for subtasks relating to the unsafe capabilities instead. Reach out to us if you are unsure on this point.

### Recommended Standards

In addition to the required standards, we recommend following the additional guidelines, noting that these are not required:

#### Scoring

We recommend your task include a binary success metric as its main scorer, representing a pass or fail on the task.  

In addition to this, we recommend the inclusion of auxiliary scorers that can provide more information rich metrics for evaluations beyond a simple binary success metric.

Finally, we ask that you restrict your use of model graded scoring to content matching between generated answers and ground truth answers. We discourage using model grading for other scoring techniques (for example asking a model to numerically rate outputs), as these approaches have known [accuracy and bias issues](https://arxiv.org/pdf/2404.13076).

#### Solver Implementation

Inspect evaluations require a [solver](https://inspect.ai-safety-institute.org.uk/solvers.html) (in our case, typically also an [agent](https://inspect.ai-safety-institute.org.uk/agents.html)) that will run the task. By default, you should use the [basic agent solver](https://github.com/UKGovernmentBEIS/inspect_ai/blob/56e8e52f9eb7b3f0b3c9093d33b6b4325ecd7e67/src/inspect_ai/solver/_basic_agent.py#L49) provided by inspect and corresponding tools. If a custom solver and/or custom tools would materially improve performance on the evaluation, you should build one and place it in `solver.py` in the root folder.

#### Setup and Dependencies

The evaluation should require minimal extra dependencies beyond those already in the template repo, and the versions of any dependencies required must be pinned.  

If using a Docker sandbox environment, you must pass an absolute path to the docker compose file in your code (see the [example](https://github.com/UKGovernmentBEIS/as-task-standard/blob/main/example_evaluations/template/template.py) in our template task), and should pin all software versions within the container.

The evaluation should not require any manual setup steps. If you absolutely need to do this, include details about it in the `README.md`, or in `CONTRIBUTING.md` as appropriate, and we will work to automate this internally.

#### Additional Testing

End-to-end tests for your evaluation that verify at least a single sample can be completed, ideally using mocked LLM API calls, are highly recommended. Here is an example of a mocked end-to-end test in [our template repository](https://github.com/UKGovernmentBEIS/as-evaluation-standard/blob/ebf3937bb7ea7612d5d1681f8676f017139b5c09/example_evaluations/web_multihop_question_answering/tests/test_web_multihop_question_answering.py#L37).

#### Observable from Logs

The decisions your agent makes and the actions it takes during the evaluation trajectory should be easily observable in the Inspect logs. To aid with this, you should be taking advantage of features from [Inspect’s Agents API](https://inspect.ai-safety-institute.org.uk/agents-api.html) to increase observability, notably:

* Avoid using `TaskState.metadata` within solvers when possible - use `TaskState.store` instead.
* Liberally use both `@subtask` [^1] and `transcript` to record specific events.

#### Difficulty

The evaluation should be challenging for current models, but theoretically possible. We should have some confidence that the evaluation is challenging enough that it won’t be saturated in the next 12-24 months. Evaluations with parametrizable difficulty or various difficulty scales are ideal.

[^1]: Note that Inspect’s `@subtask` is distinct from how we refer to subtasks in this document. The Inspect concept refers to an isolation mechanism for parts of the evaluation implementation logic, whereas we refer to subtasks as a way of breaking down an evaluation into multiple smaller evaluations.

#### Consistency

Ideally, we want evaluations to behave the same every time we run them, and for the scores not to change over time given the same model. If evaluations rely on resources that are liable to change over time (e.g. websites or services), consider whether it’s possible to mock these, or to replace them with a static version of the resource.

#### Specificity and Robustness

Evaluations should aim to measure one capability specifically and to measure that one capability robustly - like a [single responsibility principle](https://en.wikipedia.org/wiki/Single-responsibility_principle) for evaluations. Try to avoid any extraneous components in your evaluation that might cause underestimation of a model’s capabilities on the task for reasons that aren’t relevant to that capability. For example, an overly strict response format could cause us to underestimate a model’s success rate.

#### Subtasks for long horizon evaluations

We recommend breaking down long horizon evaluations into subtasks: For evaluations involving multiple independent steps, consider defining separate subtasks for each significant step. It's advisable to include a single end-to-end task in addition to individual subtasks. If you choose this approach, detail each task separately in METADATA.json, using a list of metadata objects instead of a single dict, and specify task relations in the parent_tasks and subtasks fields.

#### Expert Consultation

It’s important that you consult an expert in the field to get an idea of how they would complete a given task.  

Without doing so you may end up implementing logic for your evaluation that is irrelevant or that gives an inaccurate picture of the task difficulty. Experts tend to have custom tools and processes that may reduce the difficulty of the task once you are aware of them, and you risk severely underestimating an agent’s performance on a task if you fail to ask, “How would an expert do this?”.

## Review Process

Once you have an evaluation prepared with all the components above, follow the steps outlined below to test it, and submit it to our evaluation suite:

* Make a [private copy](https://docs.github.com/en/repositories/creating-and-managing-repositories/duplicating-a-repository) of the [as-evaluation-standard](https://github.com/UKGovernmentBEIS/as-evaluation-standard/) repository.
* Install the package and then create a new branch of the Git repo for your evaluation.
* Then, run the test runner on your own evaluation (see “Running tests on your evaluation” in README.md).
* Iterate on your evaluation until all tests pass – if anything is unclear, feel free to reach out to a developer at AISI for help.
* When your evaluation passes all the tests, invite a reviewer from AISI to your private fork.
* Start the review process by opening a PR from the evaluation branch of your private fork to the main branch of the private fork.
* Once reviewed and approved, we will onboard the evaluation to our private AS-evals repo.

## Definitions

* **Evaluation**: A test to assess specific AI capabilities, consisting of variants or categories, instructions, input data, and scoring functions.  
* **Variant**: A single instance or sample of an evaluation.  
* **Category**: A natural grouping of variants within an evaluation.  
* **Task**: In Inspect, a runnable evaluation. A full evaluation may be defined by multiple inspect task objects, each of which may represent a single category, for example.  
* **Subtask**: A smaller, self-contained unit of work within a larger evaluation. Subtasks could be considered to be smaller evaluations. Subtasks can be used to break down complex evaluations into more manageable pieces, allowing for more granular assessment of an agent's capabilities.
