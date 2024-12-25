if __name__ == "__main__":
    import logging

    import torch
    import torchvision.models as tv_mdl
    from delegator import FunctionProfiler
    from profiler import BasicProfiler

    def compute_gflops(prof):
        events = prof.events()
        total_flops = (
            sum([int(evt.flops) for evt in events if isinstance(evt.flops, int)])
            / 1024**3
        )
        print(f"Total GFLOPs: {total_flops:.3f} GFLOPs")

    logging.basicConfig(
        level="DEBUG", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    profiler = BasicProfiler(name="foo", profile_time=True, extra=True)
    profiler = FunctionProfiler(profiler=profiler)
    net = profiler(tv_mdl.resnet101().to(device="cuda"))
    loss_fn = profiler(torch.nn.MSELoss())

    # net = torch.nn.Linear(12, 2).to(device='cuda')
    inp1 = torch.rand(20, 3, 224, 224, device="cuda")
    inp2 = torch.rand(20, 3, 224, 224, device="cuda")
    trg = torch.rand(20, 1000, device="cuda")

    loss1 = loss_fn(net(inp1), trg)
