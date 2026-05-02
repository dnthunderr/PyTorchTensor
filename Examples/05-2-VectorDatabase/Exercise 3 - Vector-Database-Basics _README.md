# Vector Database Basics

Vector databases help us store, manage, and query the embeddings we created for generative AI, recommenders, and search engines.

Across many of the common use cases, users often find that they need to manage more than just vectors.
To make it easier for practitioners, vector databases should store and manage all of the data they need:
- embedding vectors
- categorical metadata
- numerical metadata
- timeseries metadata
- text / pdf / images / video / point clouds

And support a wide range of query workloads:
- Vector search (may require ANN-index)
- Keyword search (requires full text search index)
- SQL (for filtering)

For this exercise we'll use LanceDB since it's open source and easy to setup.

## Creating tables and adding data

Let's create a LanceDB table called `cats_and_dogs` under the local database directory `~/.lancedb`.
This table should have 4 fields:
- the embedding vector
- a string field indicating the species (either "cat" or "dog")
- the breed
- average weight in pounds

We're going to use pydantic to make this easier. 
First we create the pydantic model, then we use it to generate the schema for the table.
We'll also use it to make it easier to create rows of data.

## Adding data

For this problem, we will connect to a local db at ~/.lancedb and 
create an empty LanceDB table called "cats_and_dogs". We then 
add some cats and add some dogs. Now we're ready to query the table.

## Querying tables

Vector databases allow us to retrieve data for generative AI applications. Let's see how that's done.

Let's say we have a new animal that has embedding [10.5, 10.], would you expect the most similar animal will be?
Can you use the table we created above to answer the question?

**HINT** you'll need to use the `search` API for LanceTable and `limit` / `to_df` APIs. For examples you can refer to [LanceDB documentation](https://lancedb.github.io/lancedb/basic/#how-to-search-for-approximate-nearest-neighbors).

Now what if we use cosine distance instead? Would you expect that we get the same answer? Why or why not?

**HINT** you can add a call to `metric` in the call chain

## Filtering tables

In practice, we often need to specify more than just a search vector for good quality retrieval. Oftentimes we need to filter the metadata as well.

Please write code to retrieve two most similar examples to the embedding [10.5, 10.] but only show the results that is a cat.

**HINT** In LanceDB, for additional filtering, you can add a call to `where` in the call chain and pass in a SQL-like filter string.


## Creating ANN indices

For larger tables (e.g., >100K rows), searching through all of the vectors becomes quite slow. Here is where the Approximate Nearest Neighbor (ANN) index comes into play. While there are many different ANN indexing algorithms, they all have the same purpose - to drastically limit the search space as much as possible while losing as little accuracy as possible

For this problem we will create an ANN index on a LanceDB table and see how that impacts performance

### First let's create some data

We'll create 100K vectors with 128D in a new table. Here the embedding values don't matter, so we simply generate random embeddings as a 2D numpy array. We then use the `vec_to_table` function to convert that in to an Arrow table, which can then be added to the table.

### Let's establish a baseline without an index

Before we create the index, let's make sure know what we need to compare against.

We'll generate a random query vector and record it's value in the `query` variable so we can use the same query vector with and without the ANN index.

Now please write code to compute the average latency of this query

### Now let's create an index

There are many possible index types ranging from hash based to tree based to partition based to graph based.
For this task, we'll create an IVFPQ index (partition-based index with product quantization compression) using LanceDB.

Please create an IVFPQ index on the LanceDB table such that each partition is 4000 rows and each PQ subvector is 8D.

**HINT** 
1. Total vectors / number of partitions = number of vectors in each partition
2. Total dimensions / number of subvectors = number of dimensions in each subvector


Now let's search through the data again. Notice how the answers now appear different.
This is because an ANN index is always a tradeoff between latency and accuracy.

Now write code to compute the average latency for querying the same table using the ANN index.

**SOLUTION** The index is implementation detail, so it should just be running the same code as above. You should see almost an order of magnitude speed-up. On larger datasets, this performance difference should be even more pronounced.


## Deleting rows

Like with other kinds of databases, you should be able to remove rows from the table.
Let's go back to our tables of cats and dogs

Can you use the `delete` API to remove all of the cats from the table?

**HINT** use a SQL like filter string to specify which rows to delete from the table


## What if I messed up?

Errors is a common occurrence in AI. What's hard about errors in vector search is that oftentimes a bad vector doesn't cause a crash but just creates non-sensical answers. So to be able to rollback the state of the database is very important for debugging and reproducibility

So far we've accumulated 4 actions on the table:
1. creation of the table
2. added cats
3. added dogs
4. deleted cats

What if you realized that you should have deleted the dogs instead of the cats?

Here we can see the 4 versions that correspond to the 4 actions we've done


```python
table.list_versions()
```




    [{'version': 1,
      'timestamp': datetime.datetime(2023, 7, 30, 11, 27, 31, 224950),
      'metadata': {}},
     {'version': 2,
      'timestamp': datetime.datetime(2023, 7, 30, 11, 27, 32, 182594),
      'metadata': {}},
     {'version': 3,
      'timestamp': datetime.datetime(2023, 7, 30, 11, 27, 33, 651013),
      'metadata': {}},
     {'version': 4,
      'timestamp': datetime.datetime(2023, 7, 30, 11, 57, 16, 274778),
      'metadata': {}}]



Please write code to "checkout" the version still containing the whole dataset


Now write code to remove the dogs instead of the cats



## Dropping a table

You can also choose to drop a table, which also completely removes the data.
Note that this operation is not reversible.

Write code to irrevocably remove the table "cats_and_dogs" from the database

How would you verify that the table has indeed been deleted?


## Summary

Congrats, in this exercise you've learned the basic operations of vector databases from creating tables, to adding data, and to querying the data. You've learned how to create indices and you saw first hand how it changes the performance and the accuracy. Lastly, you've learned how to debug and rollback when errors happen.
