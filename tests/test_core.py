from tcg_rl.encoding import EncoderConfig,encode_observation
from ptcg_ppo.decks import read_deck
from opponents import load_opponent

def test_decks_have_60_cards():
    assert len(read_deck("decks/kangaskhan_multitype.csv"))==60
    assert len(load_opponent("mega_lucario").deck)==60
    assert len(load_opponent("mega_starmie").deck)==60
