from ge_act.experiments.robofollow_policy import cached_text_conditioning


def test_cache_is_bounded_and_keyed_by_actual_instruction():
    class Pipeline:
        def __init__(self):
            self.calls = []

        def encode_prompt(self, **kwargs):
            self.calls.append(kwargs)
            return object(), object(), None, None

    pipe = Pipeline()
    encode = cached_text_conditioning(pipe)
    left = encode('left')
    assert encode('left') is left
    assert encode('right') is not left
    assert encode('') is not left
    assert encode('left') is not left
    assert [c['prompt'] for c in pipe.calls] == ['left', 'right', '', 'left']
    assert all(c['do_classifier_free_guidance'] is False for c in pipe.calls)
    assert encode.cache_info().currsize == 1
    assert set(left) == {'prompt_embeds', 'prompt_attention_mask',
                         'negative_prompt_embeds', 'negative_prompt_attention_mask'}
    assert cached_text_conditioning(Pipeline())('left') is not left
