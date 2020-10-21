import torch as th


def test_item_self_similarity(initialized_model):
    model = initialized_model

    model.use_biases = False
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_items(query_ids)

    assert (ids[:, 0] == query_ids).all()

    model.use_biases = True
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_items(query_ids)

    assert (ids[:, 0] == query_ids).all()


def test_item_self_similarity_after_training(trained_model):
    model = trained_model

    model.use_biases = False
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_items(query_ids)

    assert (ids[:, 0] == query_ids).all()

    model.use_biases = True
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_items(query_ids)

    assert (ids[:, 0] == query_ids).all()


def test_user_item_similarity_after_training(trained_model):
    model = trained_model

    model.use_biases = False
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_users(query_ids)

    assert (ids[:, 0] == query_ids).all()

    model.use_biases = True
    query_ids = th.tensor(list(range(100)))
    _, ids = model.similar_to_users(query_ids)

    assert (ids[:, 0] == query_ids).all()
