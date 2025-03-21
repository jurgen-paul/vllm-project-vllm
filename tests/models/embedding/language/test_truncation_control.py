# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm.entrypoints.llm import LLM

llm = LLM(model="intfloat/multilingual-e5-large", task="embed")

small_input_str = "In which state the capital of the United States is located?"

big_input_str = "Immerse yourself in the enchanting \
    chronicle of calculus, a \
    mathematical domain that has radically transformed \
    our comprehension of change \
    and motion. Despite its roots in ancient civilizations, \
    the formal birth of \
    calculus predominantly occurred in the 17th century, \
    primarily under the \
    influential guidance of Sir Isaac Newton and\
    Gottfried Wilhelm Leibniz.\
    The earliest traces of calculus concepts are found \
    in ancient Greek mathematics,\
    most notably in the works of Eudoxus and Archimedes, \
    around 300 BCE. They utilized \
    the 'method of exhaustion'—a technique for \
    computing areas and volumes through the \
    use of finite sums. This methodology laid crucial \
    foundational work for integral \
    calculus. In the 17th century, both Newton and \
    Leibniz independently pioneered \
    calculus, each contributing unique perspectives \
    that would shape this new field.\
    Sir Isaac Newton (1642-1727) introduced calculus-like\
         methods in his seminal work,\
    Mathematical Principles of Natural Philosophy, \
    published in 1687. These methods were\
    instrumental in explaining celestial mechanics \
    and universal gravitation, featuring \
    'fluxions' (rates of change) and 'fluents' (variable quantities) \
    that echoed\
    derivatives and integrals. Gottfried Wilhelm Leibniz, \
    working independently\
    yet concurrently, made a parallel yet distinct contribution. \
    His 1684 publication \
    not only introduced notation still used today—including \
    symbols for derivatives\
    (d/dx) and integrals (∫)—but also offered a more \
    general and symbolic approach to\
    calculus, thereby widening its applicability across \
    various fields. Throughout the\
    18th and 19th centuries, mathematicians such as \
    Leonhard Euler, Joseph-Louis Lagrange,\
     and Augustin-Louis Cauchy further refined and extended \
    calculus. Euler's expansive body\
    of work encompassed differential equations and the \
    calculus of variations\
    , among others. A pivotal moment in the formalization \
    of calculus came\
     with Augustin-Louis Cauchy's articulation of the concept of limits in \
    the 19th century. This development provided a rigorous \
    mathematical foundation\
    for the infinitesimal methods used in calculus, thus \
    enhancing its theoretical\
    underpinnings. In the 19th and 20th centuries, \
    calculus spread its influence\
    far and wide, becoming an indispensable tool \
    in fields as diverse as physics,\
    engineering, economics, and computer science. Its \
    capacity to quantify change\
    and motion has been instrumental in propelling \
    technological advancements and\
    scientific discoveries that characterize our modern \
    world. The historical \
    journey of calculus—from its ancient origins to its contemporary \
    applications—illustrates a discipline\
    continuously enriched and expanded\
    by generations of mathematicians. \
    Calculus's power to encapsulate the essence\
    of change and motion has been critical in driving innovation."


def test_smaller_truncation_size(llm=llm,
                                 input_str=small_input_str,
                                 truncate_prompt_tokens=10):

    llm_output = llm.encode(input_str,
                            truncate_prompt_tokens=truncate_prompt_tokens)

    tokenizer = llm.llm_engine.get_tokenizer()

    tokenized_input = tokenizer.encode(text=input_str,
                                       truncation=True,
                                       max_length=truncate_prompt_tokens)
    prompt_tokens = llm_output[0].prompt_token_ids

    assert len(prompt_tokens) == len(tokenized_input)


def test_bigger_truncation_size(llm=llm,
                                input_str=big_input_str,
                                truncate_prompt_tokens=513):

    with pytest.raises(ValueError):
        assert str(
            llm.encode(input_str,
                       truncate_prompt_tokens=truncate_prompt_tokens)
        ) == "truncate_prompt_tokens value (513)\
                                    is greater than max_model_len (512). \
                                    Please, select a smaller truncation size."


def test_max_truncation_size(llm=llm,
                             input_str=big_input_str,
                             truncate_prompt_tokens=-1):

    llm_output = llm.encode(input_str,
                            truncate_prompt_tokens=truncate_prompt_tokens)

    prompt_tokens = llm_output[0].prompt_token_ids

    assert llm.llm_engine.model_config.max_model_len == len(prompt_tokens)
