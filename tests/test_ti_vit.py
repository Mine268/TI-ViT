from functools import partial
import cv2
import torch
from torchvision import transforms
from einops import reduce

from sl_vit2.net import TI_ViT
from sl_vit2.net.latent_transformers import TransformerBlock
from sl_vit2.net.latent_transformers import ContinuousAngleEmbedding
from sl_vit2.net.latent_transformers import ImageLatentTransformerGroup


def test_load_model():
    net = TI_ViT("./models/facebook/vit-mae-base")
    net = net.to("cuda:0")

    # class1 = type(net)

    # net = TI_ViT()
    # net = net.to("cuda:0")

    # class2 = type(net)

    # assert(class1 == class2)


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
    net = TI_ViT("./models/facebook/converted-vit-base").cuda(0)
    x = torch.zeros(size=[2,3,224,224]).cuda(0)

    print(net.forward(x, False))
    print(net.forward(x, True))


def test_TI_ViT_not_collapse():
    device = torch.device("cpu")
    model = TI_ViT(
        backbone_ckpt_dir="models/facebook/converted-vit-base",
        backbone_arch_path=None,
        decoder_ckpt_path="models/facebook/vit-mae-base-decoder/vit-mae-base-decoder.pth",
        decoder_arch_path="models/facebook/vit-mae-base-decoder/config.json",
    )
    # peft-lize
    model = TI_ViT.setup_lora_model(
        model,
        backbone_target_modules=["query", "key", "value"],
        backbone_lora_rank=1,
        decoder_target_modules=["value"],
        decoder_lora_rank=1,
    )
    model.load_state_dict(
        torch.load("checkpoints/pretrain_ego4d_20250313_1/checkpoint.pt")["model"]
    )
    model = TI_ViT.merge_lora_model(model)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=[224, 224])
    ])

    img_bgr1 = cv2.imread(
        "/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1/images/val/Capture0/ROM01_No_Interaction_2_Hand/cam400262/image21511.jpg"
    )
    img_rgb1 = cv2.cvtColor(img_bgr1, cv2.COLOR_BGR2RGB)
    img_tensor1 = preprocess(img_rgb1).unsqueeze(0)
    img_tensor1_hf = torch.flip(img_tensor1, dims=[-1])
    patches1 = model.encode(img_tensor1.to(device))
    hf_patches1 = model.encode(img_tensor1_hf.to(device))
    patches1_hf = model.trans_grp.do_hf(patches1)

    img_bgr2 = cv2.imread(
        "/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1/images/val/Capture0/ROM03_LT_No_Occlusion/cam400268/image16106.jpg"
    )
    img_rgb2 = cv2.cvtColor(img_bgr2, cv2.COLOR_BGR2RGB)
    img_tensor2 = preprocess(img_rgb2).unsqueeze(0)
    patches2 = model.encode(img_tensor2.to(device))

    def patch_delta(ps1: torch.Tensor, ps2: torch.Tensor) -> float:
        return reduce((ps1 - ps2).abs(), "b l d -> b", reduction="mean").mean()

    print(f"\nPatch norm for image1: {patch_delta(patches1, torch.zeros_like(patches1))}")
    print(f"Patch norm for flipped image 1: {patch_delta(hf_patches1, torch.zeros_like(hf_patches1))}")
    print(f"Norm for flipped patch of image 1: {patch_delta(patches1_hf, torch.zeros_like(patches1_hf))}")
    print(f"Patch norm for image2: {patch_delta(patches2, torch.zeros_like(patches2))}")
    print("---")
    print(f"Diff norm between flipped image: {patch_delta(patches1, hf_patches1)}")
    print(f"Diff norm between flipped patch: {patch_delta(patches1, patches1_hf)}")
    print(f"Diff norm between different images: {patch_delta(patches1, patches2)}")
