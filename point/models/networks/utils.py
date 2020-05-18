import torch

def aggregation_encode(output_wh):
    batch, _, w, h = output_wh.size()
    x = torch.linspace(0, w - 1, w).repeat(batch, h, 1)
    y = torch.transpose(torch.linspace(0, h - 1, h).repeat(batch, w, 1), 1, 2)
    rtx = x - output_wh[:, 0, :, :]/2
    rty = y - output_wh[:, 1, :, :]/2
    rbx = x - output_wh[:, 0, :, :]/2
    rby = y + output_wh[:, 1, :, :]/2
    ltx = x + output_wh[:, 0, :, :]/2
    lty = y - output_wh[:, 1, :, :]/2
    lbx = x + output_wh[:, 0, :, :]/2
    lby = y + output_wh[:, 1, :, :]/2

    corner = torch.cat((rtx, rty, rbx, rby, ltx, lty, lbx, lby), 1)
    return corner