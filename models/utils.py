import torch

def box_iou(boxes1, boxes2):
    # boxes: (N,4) xyxy
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(0) * (boxes1[:,3]-boxes1[:,1]).clamp(0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(0) * (boxes2[:,3]-boxes2[:,1]).clamp(0)

    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:,None] + area2 - inter
    return inter / (union + 1e-6)

def encode_boxes(anchors, gt):
    # anchors, gt: (N,4) xyxy
    ax = (anchors[:,0] + anchors[:,2]) / 2
    ay = (anchors[:,1] + anchors[:,3]) / 2
    aw = (anchors[:,2] - anchors[:,0]).clamp(min=1e-6)
    ah = (anchors[:,3] - anchors[:,1]).clamp(min=1e-6)

    gx = (gt[:,0] + gt[:,2]) / 2
    gy = (gt[:,1] + gt[:,3]) / 2
    gw = (gt[:,2] - gt[:,0]).clamp(min=1e-3)
    gh = (gt[:,3] - gt[:,1]).clamp(min=1e-3)


    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)
    return torch.stack([tx, ty, tw, th], dim=1)

def decode_boxes(anchors, deltas):
    ax = (anchors[:,0] + anchors[:,2]) / 2
    ay = (anchors[:,1] + anchors[:,3]) / 2
    aw = (anchors[:,2] - anchors[:,0]).clamp(min=1e-6)
    ah = (anchors[:,3] - anchors[:,1]).clamp(min=1e-6)

    tx, ty, tw, th = deltas.unbind(dim=1)
    gx = tx * aw + ax
    gy = ty * ah + ay
    gw = torch.exp(tw) * aw
    gh = torch.exp(th) * ah

    x1 = gx - gw/2
    y1 = gy - gh/2
    x2 = gx + gw/2
    y2 = gy + gh/2
    return torch.stack([x1,y1,x2,y2], dim=1)
