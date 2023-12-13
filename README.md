# CS 6200 / IS 4200 Final Project

Note that this relies on [this](https://github.com/ramybaly/Article-Bias-Prediction) repo to be
installed somewhere, and currently you have to manually point to it in code. In the future this
should be automatic. The `data/` folder is not pushed as it's very large and slow (just like the
repo linked).

## Grade Contract / Goals
**B**:  A Naive Bayes model to predict political sentiment, trained on above annotated data.
- [x] Document parsing
- [x] Training
- [x] Test accuracy

**B+**:  Multiple naive Bayes implementations with either some word modification/removal/n-grams and
a comparison between the models.
- [x] stemming
- [x] stopping
- [ ] Bigram model
- [ ] Weighted bigram? like bigram but fallback to unigram model?

**A-**:  Create a scale out of the naive Bayes results, so that articles can be deemed "more right"
than other right learning articles (similar with left leaning articles). Given that there are levels
of bias, different articles could be more or less biased than others, and being able to predict that
is useful. Maybe someone searching for news is okay with a slightly right leaning article, but wants
to stay away from conspiracy theories, or something akin to propaganda.
- [ ] Judge Bias on a scale instead of a trinary

**A**:  Use our model results to augment an existing search engine. This can be done in a few
different ways, and ideally there will be options, but minimally, articles with little to no bias
(deemed "center") will be boosted in the results list.
- [x] Tantivy Search Engine indexes docs, takes queries and shows results
- [ ] Rankings are boosted based on political sentiment
- [ ] Some type of evaluation for this model (?) I'm not sure that we have to do this. I really
don't want to

Additionally, a simple command line interface to our application which allows users to enter a query
and see results (labeled with their bias), filter results by bias, and enter text to check against
our model for bias.
- [x] CLI for querying our models (both political sentiment and searching for articles)
- [ ] Filter results by bias
