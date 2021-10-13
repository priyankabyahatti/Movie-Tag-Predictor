import pytest
from model_prediction import tokenize, predicttags


# def test_train_test_exists():
#     train, test = getprocesseddata()
#     assert len(train) and len(test) != 0, "Train and test dfs don't exist"


def test_tokenize():
    test_tags = "suspensful ,flashback, thriller"
    tags = str(test_tags)
    test_tags = str(tokenize(tags))
    assert test_tags.isspace() is not True, "tokenize function defective"


# def test_tags_output():
#     plot = '''Thirteen-year-old Becky Hooper is being questioned regarding an event that recently took place at her family's house. She gives vague answers and does not seem to remember.Two weeks earlier, Becky was a bullied high school student whose mother passed away a year ago. She has a strained relationship with her father Jeff, who attempts to reconnect with her with a trip to their lakefront home. Meanwhile, prisoner Dominick, a Neo-Nazi, and his men Apex, Cole, and Hammond are riding in a transport van. Dominick has an inmate killed to get the guards to pull over, using the opportunity to kill them and pose as policemen. They stop a man and his two children on the street and take their car, presumably killing them.Jeff's girlfriend Kayla and her young son Ty arrive at the house, upsetting Becky. Jeff announces that he and Kayla are engaged. Hurt, Becky runs out of the house, followed by one of her dogs, Diego. At her small fort in the woods, she retrieves a large key, the bow of which is formed in the shape of a valknut symbol. Dominick and his men show up at the house, take everyone hostage, and demand the key. Kayla and Ty try to escape but are caught by Apex, who tries to help them while Cole kills Dora, one of the family's dogs. Jeff lies about Becky's presence to protect her, but Dominick catches on and shoots Kayla in the leg to get the truth out of them.Becky, still in the woods, becomes aware of the intruders' presence and talks over a walkie-talkie, lying about calling the cops; Dominick calls her bluff and brings Jeff outside to the family's firepit to lure Becky out. He begins to torture Jeff with a metal rod. Becky relents and says she will give him the key. Dominick allows Jeff to talk through the talkie, but he tells Becky to run. Jeff breaks free and finds Becky, telling her he loves her before being shot dead. Dominick demands the key and Becky gouges out his left eye with it before fleeing with Diego.Dominick goes back into the house to cut off the dangling eye and sends Cole and Hammond out to retrieve the key. Cole finds and chases Becky back to her fort. Cole tries to negotiate with Becky, but she surprises him and stabs him repeatedly with colored pencils and jabs a sharp ruler through his neck, killing him.'''
#     plot = str(plot)
#     tags = predicttags(plot)
#     assert tags is not None, 'Tags not predicted'


if __name__ == '__main__':
    pytest.main()
