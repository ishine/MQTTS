from text import chinese, cleaned_text_to_sequence, symbols


def clean_text(text):
    norm_text = chinese.text_normalize(text)
    phones = chinese.g2p(norm_text)
    for ph in phones:
        assert ph in symbols
    return phones

def text_to_sequence(text):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)

if __name__ == '__main__':
    print(clean_text("你好，啊啊啊额、还是到付红四方。"))


