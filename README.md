# Are Rotten Apples Edible? Challenging Commmonsense Inference Ability with Exceptions

**By**: Nam Do ([Email](nam_do@alumni.brown.edu), [Website](https://ndo3.github.io/), [Google Scholar](https://scholar.google.com/citations?user=TgI_6s0AAAAJ)), Ellie Pavlick ([Email](ellie_pavlick@brown.edu), [Website](http://cs.brown.edu/people/epavlick/), [Google Scholar](https://scholar.google.com/citations?user=sFyrSa8AAAAJ&hl=en))

**Important links**: [Paper](https://aclanthology.org/2021.findings-acl.181.pdf), [Code and Data](https://github.com/commonsense-exception/commonsense-exception)

**BibTeX***:
```
@article{dorotten,
  title={Are Rotten Apples Edible? Challenging Commonsense Inference Ability with Exceptions},
  author={Do, Nam and Pavlick, Ellie}
}
```

## About This Repository

### Dataset

#### Loading the data
The dataset is located at `data/winoventi_bert_large_final.tsv`. When loading the data, it is important to note that the data is a tab-separated sheet (separated by `\t`). An example code to load the data:

```
imporot pandas as pd

data = pd.read_csv("data/winoventi_bert_large_final.tsv", sep="\t")
```

#### Metadata

**Dataset length**: There are 4352 rows and 9 fields in the dataset, representing 4352 challenges (2176 adversarial, 2176 stereotypical) to a language model.

**Fields**: The fields and descriptions of types and what they represent are as follows:

1. `Word`: A String, that represents the entity of interest (from the THINGS dataset, as mentioned in the paper)
2. `Associative Bias`: A String, that represents *adjectives* that are associated with the entity regardless of the context being positive or negative (e.g., `apple` is associated with `edible` regardless of the context being `The apple is _____` or `The apple is not _____`).
3. `Alternative`: A String, that represents the *crowdsourced* adjectives that might be true of the entity when the associative bias adjective is not (see paper).
4. `biased_word_context`: A String, that represents the context that makes the entity to be correctly characterized by the associative bias adjective and not by the alternative adjective. See paper for a more detailed description.
5. `adversarial_word_context`: A String, that conversely represents the context that makes the entity to be correctly characterized by the alternative adjective and not by the associative bias adjective. See paper for a more detailed description.
6. `masked_prompt`: A String that combines the context and the descriptor of the entity and mask the correct answer.
7. `target`: A String, that represents the correct answer to the `masked_prompt`.
8. `incorrect`: A String, that represents the *incorrect* andswer to the `masked_prompt`.
9. `test_type`: A number, that represents the type of challenge that the schema is testing. `1` represents the "stereotypical challenge", testing whether a language model correctly predicts the associative bias descriptor when the context is the `biased_word_context`. `2` represents the "exception challenge", testing whether the language model correctly predicts the alternative descriptor when the context is the `adversarial_word_context`.
