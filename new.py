from neo4j import GraphDatabase
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Concatenate
from tensorflow.python.keras.models import Model
import tensorflow_gnn as tfgnn

def extract_graph_data(driver):
    with driver.session() as session:
        # Extract nodes
        user_result = session.run("MATCH (user:User) RETURN user.userId, user.username,user.email")
        stream_result = session.run("MATCH (stream:Stream) RETURN stream.streamID, stream.streamName, stream.views, stream.likes, stream.comments")
        category_result = session.run("MATCH (category:Category) RETURN category.name")
        location_result=session.run("MATCH (location:Location) RETURN location.city")
        # Extract edges
        follows_result = session.run("MATCH (user:User)-[:FOLLOWS]->(follower:User) RETURN user.userId, follower.userId")
        views_result = session.run("MATCH (user:User)-[:VIEWED]->(stream:Stream) RETURN user.userId, stream.streamID")
        engages_with_result = session.run("MATCH (user:User)-[:Engages_WITH]->(stream:Stream) RETURN user.userId, stream.streamID")
        belongs_to_result = session.run("MATCH (stream:Stream)-[:BELONGS_TO]->(category:Category) RETURN stream.streamID, category.name")
        lived_in_result=session.run("MATCH (user:User)-[:LIVES_IN]->(location:Location) RETURN user.userId, location.city")

        # Create DataFrames
        user_df = pd.DataFrame([record.values() for record in user_result])
        stream_df = pd.DataFrame([record.values() for record in stream_result])
        category_df = pd.DataFrame([record.values() for record in category_result])
        location_df=pd.DataFrame([record.values() for record in location_result])
        follows_df = pd.DataFrame([record.values() for record in follows_result])
        views_df = pd.DataFrame([record.values() for record in views_result])
        engages_with_df = pd.DataFrame([record.values() for record in engages_with_result])
        belongs_to_df = pd.DataFrame([record.values() for record in belongs_to_result])
        lived_in_df=pd.DataFrame([record.values() for record in lived_in_result])
    return user_df, stream_df, category_df, location_df, follows_df, views_df, engages_with_df, belongs_to_df, lived_in_df

import tensorflow as tf
import tensorflow_gnn as tfgnn

def create_tf_graph(user_df, stream_df, category_df, follows_df, views_df, engages_with_df, belongs_to_df, lived_in_df, location_df):
    """
    Creates a TensorFlow Graph object from extracted DataFrames.

    Args:
        user_df: DataFrame containing user data.
        stream_df: DataFrame containing stream data.
        category_df: DataFrame containing category data.
        follows_df: DataFrame containing user-follow relationships.
        views_df: DataFrame containing user-view relationships.
        engages_with_df: DataFrame containing user-engagement relationships.
        belongs_to_df: DataFrame containing stream-category relationships.
        lived_in_df: DataFrame containing user-location relationships.
        location_df: DataFrame containing location data.

    Returns:
        A TensorFlow Graph object representing the extracted graph data.
    """
    
    # Convert user and stream dataframe columns to strings
    user_df = user_df.astype('str')
    stream_df = stream_df.astype('str')

    # Convert category and location dataframe
    category_df = category_df.astype('str')
    location_df = location_df.astype('str')
    lived_in_df = lived_in_df.astype('str')
    follows_df = follows_df.astype('str')
    views_df = views_df.astype('str')
    engages_with_df = engages_with_df.astype('str')
    belongs_to_df = belongs_to_df.astype('str')

    # Create node specifications for users
    user_features = {
        'userId': tf.convert_to_tensor(user_df[0].values, dtype=tf.string),  # User ID as string
        'username': tf.convert_to_tensor(user_df[1].values, dtype=tf.string),  # Username as string
        'email': tf.convert_to_tensor(user_df[2].values, dtype=tf.string)  # Email as string
    }

    # Create node specifications for streams
    stream_features = {
        'streamId': tf.convert_to_tensor(stream_df[0].values, dtype=tf.string),  # Stream ID as string
        'streamName': tf.convert_to_tensor(stream_df[1].values, dtype=tf.string),  # Stream Name as string
        'views': tf.convert_to_tensor(stream_df[2].values, dtype=tf.float32),  # Views as float32
        'likes': tf.convert_to_tensor(stream_df[3].values, dtype=tf.float32),  # Likes as float32
        'comments': tf.convert_to_tensor(stream_df[4].values, dtype=tf.float32)  # Comments as float32
    }

    # Create node specifications for categories and locations
    category_features = {'name': tf.convert_to_tensor(category_df[0].values, dtype=tf.string)}  # Category names
    location_features = {'city': tf.convert_to_tensor(location_df[0].values, dtype=tf.string)}  # Locations

    # Define node sets
    user_spec = tfgnn.NodeSet.from_fields(features=user_features, sizes=tf.constant([len(user_df)]))
    stream_spec = tfgnn.NodeSet.from_fields(features=stream_features, sizes=tf.constant([len(stream_df)]))
    category_spec = tfgnn.NodeSet.from_fields(features=category_features, sizes=tf.constant([len(category_df)]))
    location_spec = tfgnn.NodeSet.from_fields(features=location_features, sizes=tf.constant([len(location_df)]))

    follows_source = tf.convert_to_tensor(follows_df[0].values, dtype=tf.int32)
    follows_target = tf.convert_to_tensor(follows_df[1].values, dtype=tf.int32)
    
    views_source = tf.convert_to_tensor(views_df[0].values, dtype=tf.int32)
    views_target = tf.convert_to_tensor(views_df[1].values, dtype=tf.int32)
    
    engages_with_source = tf.convert_to_tensor(engages_with_df[0].values, dtype=tf.int32)
    engages_with_target = tf.convert_to_tensor(engages_with_df[1].values, dtype=tf.int32)
    
    belongs_to_source = tf.convert_to_tensor(belongs_to_df[0].values, dtype=tf.int32)

    # Create a mapping of category names to unique integers
    category_mapping = {category: idx for idx, category in enumerate(category_df[0].unique())}

    # Map the 'belongs_to' column from string to integer
    belongs_to_target = belongs_to_df[1].map(category_mapping).values

    # Convert to TensorFlow tensor with integer type
    belongs_to_target = tf.convert_to_tensor(belongs_to_target, dtype=tf.int32)
    

    location_mapping = {location: idx for idx, location in enumerate(location_df[0].unique())}
    lived_in_source = tf.convert_to_tensor(lived_in_df[0].values, dtype=tf.int32)
    lived_in_target = lived_in_df[1].map(location_mapping).values
    lived_in_target = tf.convert_to_tensor(lived_in_target, dtype=tf.int32)

    # Define the EdgeSets
    follows_spec = tfgnn.EdgeSet.from_fields(
        features={},
        sizes=tf.constant([len(follows_df)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=('user', follows_source),
            target=('user', follows_target)
        )
    )
    
    views_spec = tfgnn.EdgeSet.from_fields(
        features={},
        sizes=tf.constant([len(views_df)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=('user', views_source),
            target=('stream', views_target)
        )
    )
    
    engages_with_spec = tfgnn.EdgeSet.from_fields(
        features={},
        sizes=tf.constant([len(engages_with_df)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=('user', engages_with_source),
            target=('stream', engages_with_target)
        )
    )
    
    belongs_to_spec = tfgnn.EdgeSet.from_fields(
        features={},
        sizes=tf.constant([len(belongs_to_df)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=('stream', belongs_to_source),
            target=('category', belongs_to_target)
        )
    )
    
    lives_in_spec = tfgnn.EdgeSet.from_fields(
        features={},
        sizes=tf.constant([len(lived_in_df)]),
        adjacency=tfgnn.Adjacency.from_indices(
            source=('user', lived_in_source),
            target=('location', lived_in_target)
        )
    )
    
    # Create the TensorFlow graph
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            'user': user_spec,
            'stream': stream_spec,
            'category': category_spec,
            'location': location_spec,
        },
        edge_sets={
            'follows': follows_spec,
            'views': views_spec,
            'engages_with': engages_with_spec,
            'belongs_to': belongs_to_spec,
            'lives_in': lives_in_spec,
        }
    )
    
    return graph


# Example usage
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "12345678"))
user_df, stream_df, category_df, location_df, follows_df, views_df, engages_with_df, belongs_to_df, lived_in = extract_graph_data(driver)
tf_graph=create_tf_graph(user_df, stream_df, category_df, follows_df, views_df, engages_with_df, belongs_to_df, lived_in, location_df)
print(tf_graph)