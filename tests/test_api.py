def test_api_stub():
    from bh_vision import BHVision
    try:
        BHVision.from_pretrained("BH-Language-AI/bh-vision-2b")
        assert False, "mora baciti NotImplementedError"
    except NotImplementedError:
        pass
