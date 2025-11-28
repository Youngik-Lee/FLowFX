def estimate_slippage(volume, market_depth=1e7, spread=0.0002):
    return spread + abs(volume) / market_depth

def apply_slippage(prediction, volume):
    slip = estimate_slippage(volume)
    return prediction - slip
