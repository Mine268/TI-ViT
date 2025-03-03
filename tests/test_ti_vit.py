from functools import partial
import torch

from sl_vit2.net import TI_ViT
from sl_vit2.net.latent_transformers import TransformerBlock
from sl_vit2.net.latent_transformers import ContinuousAngleEmbedding
from sl_vit2.net.latent_transformers import ImageLatentTransformerGroup


def test_load_model():
    net = TI_ViT("./models/facebook/vit-mae-base")
    net = net.to("cuda:7")

    class1 = type(net)

    net = TI_ViT()
    net = net.to("cuda:7")

    class2 = type(net)

    assert(class1 == class2)


def test_TransformerBlock():
    block = TransformerBlock(dim=768, num_heads=12).cuda(0)
    x = torch.randn(4, 33, 768).cuda(0)

    y = block(x)
    print(y.shape)


def test_ContinousAngleEmbedding():
    embedder = ContinuousAngleEmbedding()
    angles = torch.tensor([0, 1.2, 0.7, 3.0])
    embeds = embedder(angles)
    print(embeds.shape)


def test_LatentTransforms1():
    trans = ImageLatentTransformerGroup().cuda(0)
    tokens = torch.randn(4, 196, 768).cuda(0)
    angles = [3.14, 2.83, 1.3994, 0.00394]
    tokens2 = trans.do_cr(tokens, angles)
    print(tokens2.shape)


def test_LatentTransforms1_compose():
    trans_grp = ImageLatentTransformerGroup()
    hf = trans_grp.get_parameterized_hf()
    cr1 = trans_grp.get_parameterized_cr([1.0, 2.0])
    hr1 = trans_grp.get_parameterized_hr([-2.0, -1.0])
    cr2 = trans_grp.get_parameterized_cr([-1.0, 2.0])
    hr2 = trans_grp.get_parameterized_hr([-2.0, 1.0])

    assert(trans_grp.compose(hf, hf).func == trans_grp.do_cr and \
        trans_grp.compose(hf, hf).keywords["angle_rad"] == None)
    assert(trans_grp.compose(hf, cr2).func == trans_grp.do_hr and \
        trans_grp.compose(hf, cr2).keywords["angle_rad"].allclose(torch.tensor([-1.0, 2.0])))
    assert(trans_grp.compose(hf, hr2).func == trans_grp.do_cr and \
        trans_grp.compose(hf, hr2).keywords["angle_rad"].allclose(torch.tensor([-2.0, 1.0])))

    assert(trans_grp.compose(cr1, hf).func == trans_grp.do_hr and \
        trans_grp.compose(cr1, hf).keywords["angle_rad"].allclose(torch.tensor([-1.0, -2.0])))
    assert(trans_grp.compose(cr1, cr2).func == trans_grp.do_cr and \
        trans_grp.compose(cr1, cr2).keywords["angle_rad"].allclose(torch.tensor([0.0, 4.0])))
    assert(trans_grp.compose(cr1, hr2).func == trans_grp.do_hr and \
        trans_grp.compose(cr1, hr2).keywords["angle_rad"].allclose(torch.tensor([-3.0, -1.0])))

    assert(trans_grp.compose(hr1, hf).func == trans_grp.do_cr and \
        trans_grp.compose(hr1, hf).keywords["angle_rad"].allclose(torch.tensor([2.0, 1.0])))
    assert(trans_grp.compose(hr1, cr2).func == trans_grp.do_hr and \
        trans_grp.compose(hr1, cr2).keywords["angle_rad"].allclose(torch.tensor([-3.0, 1.0])))
    assert(trans_grp.compose(hr1, hr2).func == trans_grp.do_cr and \
        trans_grp.compose(hr1, hr2).keywords["angle_rad"].allclose(torch.tensor([0.0, 2.0])))


def test_TI_ViT_forward_loss():
    net = TI_ViT("./models/facebook/converted-vit-base").cuda(7)
    x = torch.zeros(size=[2,3,224,224]).cuda(7)

    print(net.forward(x, False))
    print(net.forward(x, True))