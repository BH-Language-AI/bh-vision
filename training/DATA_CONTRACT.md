# DATA CONTRACT — image-text-to-text chat JSONL, jedan red = jedan uzorak
# Template se primjenjuje preko processor.apply_chat_template (placeholder <image>, ne ručni tokeni).
# shortcut: multi-image/video batch miješanje i max_length truncacija brišu image tokene — držati max_length=null i homogeni batch.
{"messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Šta piše na računu?"}]}], "images": ["data/img/001.jpg"]}
