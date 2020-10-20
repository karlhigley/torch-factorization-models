from hypothesis import strategies as strat


@strat.composite
def raw_predictions(draw):
    """Test case generation strategy for raw positive and negative predictions"""
    # batch_size = draw(strat.integers(min_value=1, max_value=1024))
    batch_size = 32

    positives_list = strat.lists(
        strat.floats(min_value=-100, max_value=100),
        min_size=batch_size,
        max_size=batch_size,
    )

    negatives_list = strat.lists(
        strat.floats(min_value=-100, max_value=100),
        min_size=batch_size,
        max_size=batch_size,
    )

    return (draw(positives_list), draw(negatives_list))
