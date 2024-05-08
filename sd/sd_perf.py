import sys

CORE = 4
CLUSTER = 1
CLK = 10**9
BW = 50 # 50GB/s
IMG_W = 512
IMG_H = 512
BATCH = 1
GM_MAX_SIZE = 8*2**20 # 8mb total
PRECISION = "FP16"

if len(sys.argv) == 8:
    print(sys.argv, len(sys.argv))
    CORE = int(sys.argv[1])
    CLUSTER = int(sys.argv[2])
    GM_MAX_SIZE= int(sys.argv[3])*2**20
    BW = int(sys.argv[4])
    IMG_W = int(sys.argv[5])
    IMG_H = int(sys.argv[6])
    PERCISION = sys.argv[7]
elif len(sys.argv) == 1:
    print("use default setting")
else:
    print("usage ./sd_perf.py [core/cluster] [cluster] [GM_size(MB)] [BW(GB)] [img_size_w] [image_size_h] [PERCISION(FP16/INT8)]\n")
    exit()

LATENT_SCALE_RATIO = 8
LATNET_W = IMG_W/LATENT_SCALE_RATIO
LATENT_H = IMG_H/LATENT_SCALE_RATIO

MTP_CI_STEP = 8
MTP_CO_STEP = 16
DATALEN = 2 if PRECISION == "FP16" else ( 1 if PRECISION == "INT8" else 4)
MAC = 16*16*8*2*2/(2 if PRECISION == "INT8" else 4)
COMP_POWER = CORE*CLUSTER*CLK*MAC

GM_size = 0 # gm_size

def rounded_ci_co(ci, co):
    ci_rounded = ((ci + (MTP_CI_STEP-1)) // MTP_CI_STEP ) * MTP_CI_STEP
    co_rounded = ((co + (MTP_CO_STEP-1)) // MTP_CO_STEP ) * MTP_CO_STEP
    return ci_rounded, co_rounded

def gm_overflow_check(size):
    global GM_size
    if size > GM_MAX_SIZE - GM_size:
        return True
    else:
        print("current gm used = %.2fMB, required = %.2fMB, free = %.2fMB"%(GM_size/2**20, size/2**20, (GM_MAX_SIZE-GM_size-size)/2**20))
        GM_size += size
        return False

def gm_remove(size):
    global GM_size
    if GM_size < size:
        print("[ERR] release GM size exceed remaind! current used %.2fMB, is releasing %.2fMB"%(GM_size/2**20, size/2**20))
        return False
    else:
        GM_size -= size
        print("gm size removed %.2fMB, still remained %.2fMB, free = %.2fMB"%(size/2**20, GM_size/2**20, (GM_MAX_SIZE-GM_size)/2**20))
        return True
    
class Feature():
    def __init__(
            self,
            n: int = BATCH,
            c: int = 4,
            h: int = LATENT_H,
            w: int = LATNET_W,
            len : int = None,
            dims: int = None,
            location: str = "GM",
            format: str = "nchw",
            name: str = "unamed feature"
    ):
        super().__init__()
        self.n, self.c, self.h, self.w, self.len, self.dims, self.location, self.format = (n,c,h,w,len,dims,location,format)
        self.name = name
        if self.format == "nchw":
            self.shape = (self.n,self.c,self.h,self.w)
            self.size = self.n*self.c*self.h*self.w*DATALEN
        elif self.format == "nld":
            self.shape = (n, len, dims)
            self.size = n*len*dims*DATALEN
        if "GM" in self.location and gm_overflow_check(self.size):
                self.location = "DDR"
        print("create feature %s w/ %.2fMB at %s"%(name,self.size/2**20,self.location))

    def release(
            self
    ):
        if "GM" in self.location:
            if not gm_remove(self.size):
                print("[ERR] release feature %s from GM failed!"%self.name)
            else:
                print("removed %s w/ %.2fMB from GM"%(self.name, self.size/2**20))
        else:
            print("feature %s is stored in DDR, doesn't need remove form GM"%self.name)

    def getSize(self):
        if self.format == "nchw":
            return self.n*self.c*self.h*self.w*DATALEN
        elif self.format == "nld":
            return self.n*self.len*self.dims*DATALEN
        else:
            return None

    def getShape(self):
        if self.format == "nchw":
            return (self.n,self.c,self.h,self.w)
        elif self.format == "nld":
            return (self.n, self.len, self.dims)

class WTConvKernel():
    def __init__(
            self,
            co: int = None,
            ci: int = None,
            kw: int = 3,
            kh: int = 3,
            stride : int = 1,
            padding: int = 1,
            location: str = "DDR"
    ):
        super().__init__()
        self.co, self.ci, self.kw, self.kh, self.stride, self.padding, self.location = (co, ci, kw, kh, stride, padding, location)
        self.shape = (co, ci, kw, kh, stride, padding)
        self.size = co*ci*kw*kh*DATALEN

class WTLinear():
    def __init__(
            self,
            ci: int = None,
            co: int = None,
            location: str = "DDR"
    ):
        super().__init__()
        self.ci, self.co, self.location = (ci, co, location)
        self.size = ci*co*DATALEN
        self.shape = (ci, co)

class Time():
    def __init__(
            self,
            op: str  = "conv2D",
            tag: str = None,
            ldst: int= 0,
            mac:  int= 0,
            op1_shape = None,
            op2_shape = None,
            out_shape = None
    ):
        super().__init__()
        self.op, self.tag, self.ldst, self.mac = (op, tag, ldst, mac)
        self.op1_shape, self.op2_shape, self.out_shape = (op1_shape, op2_shape, out_shape)
        self.op_time = self.ldst if self.ldst >= self.mac else self.mac
        self.bottleneck = "ldst" if self.ldst >= self.mac else "mac"
        self.precent = 0.0

    def update(
            self,
            total: float = 1.0
    ):
        self.precent = self.op_time/total*100

def formatsize(size, type="KB"):
    if type == "KB":
        scale = 2**10
    elif type == "MB":
        scale = 2**20
    elif type == "GB":
        scale = 2**30
    elif type == "B":
        scale = 1
    return size/scale
        
def op_conv(input_feature:Feature, kernel:WTConvKernel, tag:str="conv", saveto:str="GM"):
    if input_feature.format != "nchw" or input_feature.c != kernel.ci:
        print("%s invalid conv2D parameters!\n"%tag)
        exit()

    ci_rounded, co_rounded = rounded_ci_co(kernel.ci, kernel.co)

    # Calculate the output dimensions
    if kernel.kh == 1:
        kernel.padding = 0
    h_out = (input_feature.h + 2 * kernel.padding - kernel.kh) // kernel.stride + 1
    w_out = (input_feature.w + 2 * kernel.padding - kernel.kw) // kernel.stride + 1

    # Prepare output feature dimensions
    output_feature = Feature(n=input_feature.n, c=kernel.co, h=h_out, w=w_out, location=saveto, name=tag+"[OPCONV_output]")
    
    # Calculate the number of MAC operations
    macs = input_feature.n * co_rounded * h_out * w_out * ci_rounded * kernel.kw * kernel.kh
    mac_time = 10**6*macs/COMP_POWER
    ldst_size = 0
    if "GM" not in input_feature.location:
        ldst_size += input_feature.getSize()
    if "GM" not in kernel.location:
        ldst_size += kernel.size
    if "DDR" in output_feature.location:
        # calculate the write time cost.
        ldst_size += output_feature.getSize()
    ldst_time = 10**6*ldst_size/BW/2**30
    op_time = Time(op="conv2D",tag=tag,ldst=ldst_time,mac=mac_time,op1_shape=input_feature.getShape(),
                   op2_shape=kernel.shape,out_shape=output_feature.getShape())

    # Printing shapes and sizes
    print("---> running %s <---"%tag)
    print("Input  Shape:", input_feature.getShape())
    print("Input   Size: %.2fKB at %s"%(formatsize(input_feature.getSize(), "KB"),input_feature.location))
    print("Kernel Shape:", kernel.shape)
    print("Kernel  Size: %.2fKB at DDR"%formatsize(kernel.size, "KB"))
    print("MAC time = %.2fus, ldst time = %.2fus"%(mac_time, ldst_time))
    print("Output Shape:", output_feature.getShape())
    print("Output  Size: %.2fKB at %s\n"%(formatsize(output_feature.getSize(), "KB"),output_feature.location))

    return output_feature, op_time

def op_linear(input_feature:Feature, wt_linear:WTLinear, tag:str="op_linear", saveto:str="GM"):
    if input_feature.format != "nld" or wt_linear.ci != input_feature.dims:
        print("invalid linear parameters!\n")
        exit()

    dims, co_rounded = rounded_ci_co(wt_linear.ci, wt_linear.co)

    # Prepare output feature dimensions
    output_feature = Feature(n=input_feature.n, len=input_feature.len, dims=wt_linear.co, format="nld", location=saveto, name=tag+"[OPLINEAR_output]")

    # Calculate the number of MAC operations
    macs = input_feature.n * input_feature.len * dims * co_rounded
    mac_time = macs/COMP_POWER
    ldst_size = 0
    if "GM" not in input_feature.location:
        ldst_size += input_feature.getSize()
    if "GM" not in wt_linear.location:
        ldst_size += wt_linear.size
    if "DDR" in output_feature.location:
        # calculate the write time cost.
        ldst_size += output_feature.getSize()
    ldst_time = 10**6*ldst_size/BW/2**30
    op_time = Time(op="Linear",tag=tag,ldst=ldst_time,mac=mac_time,op1_shape=input_feature.getShape(),
                   op2_shape=wt_linear.shape,out_shape=output_feature.getShape())
    
    # Printing shapes and sizes
    print("---> running %s <---"%tag)
    print("Input     Shape:", input_feature.getShape())
    print("Input      Size: %.2fKB at %s"%(formatsize(input_feature.getSize(), "KB"),input_feature.location))
    print("wt_linear Shape:", wt_linear.shape)
    print("wt_linear  Size: %.2fKB at DDR"%formatsize(wt_linear.size, "KB"))
    print("MAC time = %.2fus, ldst time = %.2fus"%(mac_time, ldst_time))
    print("Output    Shape:", output_feature.getShape())
    print("Output     Size: %.2fKB at %s\n"%(formatsize(output_feature.getSize(), "KB"),output_feature.location))

    return output_feature, op_time

def op_bmm(input1:Feature, input2:Feature, tag:str="op_bmm", saveto:str="GM"):
    if input1.format != "nld" or input2.format != "nld" or input1.dims != input2.len:
        print("invalid bmm parameters!\n")
        exit()

    dims, co_rounded = rounded_ci_co(input1.dims, input2.dims)

    # Prepare output feature dimensions
    output_feature = Feature(n=input1.n, len=input1.len, dims=input2.dims, format="nld", location=saveto, name=tag+"[OPBMM_output]")

    # Calculate the number of MAC operations
    macs = input1.n * input1.len * dims * co_rounded
    mac_time = macs/COMP_POWER
    ldst_size = 0
    if "GM" not in input1.location:
        ldst_size += input1.getSize()
    if "GM" not in input2.location:
        ldst_size += input2.getSize()
    if "DDR" in output_feature.location:
        # calculate the write time cost.
        ldst_size += output_feature.getSize()
    ldst_time = 10**6*ldst_size/BW/2**30
    op_time = Time(op="bmm",tag=tag,ldst=ldst_time,mac=mac_time,op1_shape=input1.getShape(),
                   op2_shape=input2.getShape(),out_shape=output_feature.getShape())
    
    # Printing shapes and sizes
    print("---> running %s <---"%tag)
    print("Input1 Shape:", input1.getShape())
    print("Input1  Size: %.2fKB at %s"%(formatsize(input1.getSize(), "KB"), input1.location))
    print("Input2 Shape:", input2.getShape())
    print("Input2  Size: %.2fKB at %s"%(formatsize(input2.getSize(), "KB"), input2.location))
    print("MAC time = %.2fus, ldst time = %.2fus"%(mac_time, ldst_time))
    print("Output Shape:", output_feature.getShape())
    print("Output  Size: %.2fKB at %s\n"%(formatsize(output_feature.getSize(), "KB"),output_feature.location))

    return output_feature, op_time

def op_eltwise(input1:Feature, input2:Feature, tag:str="op_eltwise", saveto:str="GM"):
    ''' igonre this part
    if input1.shape != input2.shape:
        print("eltwise shape doesn't match")
        exit()
    '''
    # current is dummy operation.
    output = Feature(n=input1.n,c=input1.c,h=input1.h,w=input1.w,len=input1.len,dims=input1.dims,location="GM",format=input1.format,name=tag)
    t = Time(tag=tag)
    return output, t

def temb_init():
    temb = Feature(n=BATCH,len=1,dims=320,location="GM",format="nld",name="temb")
    wt_linear1 = WTLinear(ci=320,co=1280,location="DDR")
    temb_linear1, time_temb_linear1 = op_linear(temb,wt_linear1,tag=temb_init.__name__+" linear1_silu",saveto="GM")
    temb.release()

    wt_linear2 = WTLinear(ci=1280,co=1280,location="DDR")
    temb_linear2, time_temb_linear2 = op_linear(temb_linear1,wt_linear2,tag=temb_init.__name__+" linear2",saveto="GM")
    temb_linear1.release()
    time = (time_temb_linear1, time_temb_linear2)

    return temb_linear2, time

def ResnetBlock2D(temb:Feature, sample:Feature, ci:int, co:int, tag:str, saveoutput:bool=True):
    tag = tag + "->" + ResnetBlock2D.__name__
    # 0 step, calc temb
    # ignore silu part and add this part to temg init. temb input fixed as 1280.
    if temb is not None:
        wt_temb_linear = WTLinear(ci=1280,co=co)
        temb_output, temb_t = op_linear(temb,wt_temb_linear,tag+"->temb_linear","GM")

    # 1 step, calc sample
    # 1.1 TODO, current ignore group norm and silu
    # 1.2 conv2d
    kernel1 = WTConvKernel(co,ci)
    sample_conv1, conv1_t = op_conv(sample,kernel1,tag+"->sample conv1","GM")

    # 1.3 TODO resdiual add 
    # 2.1 TODO group+silu
    if temb is not None:
        temb_output.release()

    # 2.2 conv2
    kernel2 = WTConvKernel(co,co)
    sample_conv2, conv2_t = op_conv(sample_conv1,kernel2,tag+"->sample conv2","GM,DDR")
    sample_conv1.release()
    # 1.3 resdiual conv
    if ci != co:
        reskernel = WTConvKernel(co,ci,1,1)
        sample_res, resconv_t = op_conv(sample, reskernel,tag+"->shortcut conv", "GM")
        sample.release()
        output, res_t = op_eltwise(sample_conv2,sample_res,tag+"->sample res add")
        sample_res.release()
        sample_conv2.release()
        t = (conv1_t, conv2_t, resconv_t, res_t)
    else:
        output, res_t = op_eltwise(sample_conv2,sample,tag+"->sample res add")
        sample.release()
        sample_conv2.release()
        t = (conv1_t, conv2_t, res_t)

    if temb is not None:
        t = (temb_t, ) + t

    return output, t

def BasicTransformerBlock(txt_emb:Feature, sample:Feature, dims:int, crossdims:int, tag:str):
    ffdims = dims*4
    # 0 self attn
    # 0.1 TODO layer Norm
    wqkv = WTLinear(ci=dims,co=dims)
    q, q_t = op_linear(sample,wqkv,tag+"->selfattn_q")
    k, k_t = op_linear(sample,wqkv,tag+"->selfattn_k")
    # TODO, transpose k to kt
    k.len = k.dims
    k.dims = q.len
    qkt, qkt_t = op_bmm(q,k,tag+"->setfattn_qkt")
    q.release()
    k.release()
    # ignore scaler and softmax
    v, v_t = op_linear(sample,wqkv,tag+"->selfattn_v")
    attn, attn_t = op_bmm(qkt,v,tag+"->selfattn_score")
    qkt.release()
    v.release()
    wt_attnlinear = WTLinear(ci=dims,co=dims)
    selfattn, selfattn_t = op_linear(attn, wt_attnlinear, tag+"->selfattn_o")
    attn.release()
    # 0.2 TODO residual add
    sample.release()
    t = (q_t,k_t,v_t,qkt_t,attn_t,selfattn_t)

    # 1 calc cross attn
    wq = WTLinear(ci=dims,co=dims)
    q,q_t = op_linear(selfattn,wq,tag+"->cross_q")
    wkv = WTLinear(ci=txt_emb.dims,co=dims)
    k,k_t = op_linear(txt_emb,wkv,tag+"->cross_k")
    # TODO transpose k to kt
    temp_value = k.len
    k.len = k.dims
    k.dims = temp_value
    qkt,qkt_t = op_bmm(q,k,tag+"->selfattn_score")
    q.release()
    k.release()
    # ignore scaler and softmax
    v, v_t = op_linear(txt_emb,wkv,tag+"->cross_v")
    attn, attn_t = op_bmm(qkt,v,tag+"->cross_score")
    qkt.release()
    v.release()
    wt_attnlinear = WTLinear(ci=dims,co=dims)
    crossattn, crossattn_t = op_linear(attn, wt_attnlinear, tag+"->crossattn_o")
    attn.release()
    # 1.2 TODO residual add
    selfattn.release()
    t += (q_t,k_t,v_t,qkt_t,attn_t,crossattn_t)

    # 2 FF
    wt_up = WTLinear(ci=dims,co=ffdims)
    up, up_t = op_linear(crossattn,wt_up,tag+"->ff_up")
    gate, gate_t = op_linear(crossattn,wt_up,tag+"->ff_gate")
    crossattn.release()
    # 2.1 ignore elt mult
    gate.release()
    wt_down = WTLinear(ci=ffdims,co=dims)
    output, down_t = op_linear(up,wt_down,tag+"->ff_down")
    up.release()
    t += (up_t, gate_t, down_t)

    return output, t

def Transformer2DModel(txt_emb:Feature, sample:Feature, ci:int, co:int, tag:str):
    tag = tag + "->" + Transformer2DModel.__name__
    # 0 igonre group norm
    # 1 calc conv in
    convin_kernel = WTConvKernel(co=co,ci=ci,kw=1,kh=1)
    conv_in, conv_in_t = op_conv(input_feature=sample,kernel=convin_kernel,tag=tag+"->conv_in")

    # 2 TODO reshape time
    conv_in.release()
    trans_in = Feature(n=conv_in.n,len=conv_in.h*conv_in.w,dims=conv_in.c,location="GM",format="nld",name=tag+"->trans_in")

    # 3 calc basic transformer block
    transform, trans_t = BasicTransformerBlock(txt_emb=txt_emb,sample=trans_in,dims=co,crossdims=768,tag=tag+"->Basictransformer")

    #4 TODO reshape time
    transform.release()
    trans_out = Feature(n=transform.n,c=sample.c,h=sample.h,w=sample.w,name=tag+"trans_out")

    # 5 conv out
    convout_kernel = WTConvKernel(co=co,ci=ci,kw=1,kh=1)
    output, conv_out_t = op_conv(trans_out,convout_kernel,tag+"->conv_out")
    trans_out.release()

    # 6 TODO resdiula add
    sample.release()

    t = (conv_in_t,) + trans_t + (conv_out_t,)

    return output, t

def Downsample2D(sample:Feature, ci:int, co:int, tag:str):
    kernel = WTConvKernel(co,ci,3,3,stride=2)
    output, t = op_conv(input_feature=sample,kernel=kernel,tag=tag+Downsample2D.__name__,saveto="GM")
    sample.release()
    return output, t

def CrossAttnDownBlock2D(input:Feature, temb:Feature, txtemb:Feature, ci:int, co:int, tag:str):
    layer1_res, resb1_t = ResnetBlock2D(temb=temb,sample=input,ci=ci,co=co,tag=tag+"->layer1")
    layer1_trans, trans1_t = Transformer2DModel(txt_emb=txtemb,sample=layer1_res,ci=co,co=co,tag=tag+"->layer1")
    layer2_res, resb2_t = ResnetBlock2D(temb=temb,sample=layer1_trans,ci=co,co=co,tag=tag+"->layer2")
    layer2_trans, trans2_t = Transformer2DModel(txt_emb=txtemb,sample=layer2_res,ci=co,co=co,tag=tag+"->layer2")
    output, down_t = Downsample2D(layer2_trans,ci=co,co=co,tag=tag)
    t = resb1_t + trans1_t + resb2_t + trans2_t + (down_t,)
    return output, t

def DownBlock2D(input:Feature, temb:Feature, ci:int, co:int, tag:str):
    layer1, res1_t = ResnetBlock2D(temb,input,ci,co,tag=tag+"->layer1")
    output, res2_t = ResnetBlock2D(temb,layer1,ci,co,tag=tag+"->layer2")
    t = res1_t + res2_t
    return output, t

def Upsample2D(input:Feature, ci:int, co:int, tag:str):
    upinput = Feature(input.n,input.c,input.h*2,input.w*2,name=tag+"upinput")
    input.release()
    kernel = WTConvKernel(ci,co)
    output, t = op_conv(upinput,kernel,tag+Upsample2D.__name__)
    upinput.release()

    return output, (t,)

def UpBlock2D(input:Feature, temb:Feature, ci:int, co:int, tag:str,resconcat):
    # TODO add resconcat to replace fix code *2
    input.c *= 2
    layer1, res1_t = ResnetBlock2D(temb,input,ci*2,co,tag=tag+"->layer1",saveoutput=False)
    layer1.c *= 2
    layer2, res2_t = ResnetBlock2D(temb,layer1,ci*2,co,tag=tag+"->layer2",saveoutput=False)
    layer2.c *= 2
    layer3, res3_t = ResnetBlock2D(temb,layer2,ci*2,co,tag=tag+"->layer3",saveoutput=False)

    output, up_t = Upsample2D(layer3,ci,co,tag+"Upsample2D")
    t = res1_t + res2_t + res3_t + up_t
    return output, t

def CrossAttnUpBlock2D(input:Feature, temb:Feature, txtemb:Feature, ci:int, co:int, tag:str, isLast:bool=False, resconcat=None):
    # concatenate comes from DDR, should add ddr reading, while compare with mac, ld time is too less. so ignore.
    input.c += resconcat[0]
    layer1_res, resb1_t = ResnetBlock2D(temb=temb,sample=input,ci=input.c,co=co,tag=tag+"->layer1",saveoutput=False)
    layer1_trans, trans1_t = Transformer2DModel(txt_emb=txtemb,sample=layer1_res,ci=co,co=co,tag=tag+"->layer1")
    layer1_trans.c += resconcat[1]
    layer2_res, resb2_t = ResnetBlock2D(temb=temb,sample=layer1_trans,ci=layer1_trans.c,co=co,tag=tag+"->layer2",saveoutput=False)
    layer2_trans, trans2_t = Transformer2DModel(txt_emb=txtemb,sample=layer2_res,ci=co,co=co,tag=tag+"->layer2")
    layer2_trans.c += resconcat[2]
    layer3_res, resb3_t = ResnetBlock2D(temb=temb,sample=layer2_trans,ci=layer2_trans.c,co=co,tag=tag+"->layer3",saveoutput=False)
    layer3_trans, trans3_t = Transformer2DModel(txt_emb=txtemb,sample=layer3_res,ci=co,co=co,tag=tag+"->layer3")

    t = resb1_t + trans1_t + resb2_t + trans2_t + resb3_t + trans3_t
    if not isLast:
        output, up_t = Upsample2D(input=layer3_trans,ci=co,co=co,tag=tag)
        t += up_t
        return output, t
    else:
        return layer3_trans,t

def UnetMidBlock2DCrossAttn(input:Feature,temb:Feature,txtemb:Feature, ci:int, co:int, tag:str):
    layer1, resb1_t = ResnetBlock2D(temb=temb,sample=input,ci=ci,co=co,tag=tag+"->layer1",saveoutput=False)
    trans, trans_t = Transformer2DModel(txt_emb=txtemb,sample=layer1,ci=ci,co=co,tag=tag+"trans")
    output, resb2_t = ResnetBlock2D(temb=temb,sample=trans,ci=ci,co=co,tag=tag+"->layer2",saveoutput=False)
    t = resb1_t + trans_t + resb2_t
    return output, t

def Unet():
    # 0 step, para init, time embedding init
    time = ()
    time_emb, t = temb_init()
    time += t
    text_emb = Feature(n=BATCH,len=77,dims=768,location="GM",format="nld",name="text emb")

    # 1st step, pre-process
    latent = Feature(n=BATCH,c=4,h=LATENT_H,w=LATNET_W,format="nchw",location="DDR",name="latent")
    conv_in = WTConvKernel(co=320,ci=4)
    # generated input will be stored in GM for next layer, and DDR for up block layer.
    input, t = op_conv(latent, conv_in, "pre-process conv_in","GM,DDR")
    time += (t,)
    # latent.release() latent will be created in DDR, doesn't need release GM size.

    # 2nd step, down block
    db1, db1_t = CrossAttnDownBlock2D(input=input,temb=time_emb,txtemb=text_emb,ci=320,co=320,tag="DB1")
    db2, db2_t = CrossAttnDownBlock2D(input=db1, temb=time_emb, txtemb=text_emb,ci=320,co=640,tag="DB2")
    db3, db3_t = CrossAttnDownBlock2D(input=db2,temb=time_emb,txtemb=text_emb,ci=640,co=1280,tag="DB3")
    downblock, db4_t = DownBlock2D(input=db3,temb=time_emb,ci=1280,co=1280,tag="DB4")

    # 3nd step, mid block
    midblock, mid_t = UnetMidBlock2DCrossAttn(input=downblock,temb=time_emb,txtemb=text_emb,ci=1280,co=1280,tag="midB")

    # 4th step, up block
    # up block concat feature parameters
    upconcat = [[1280,1280,1280],[1280,1280,640],[640,640,320],[320,320,320]]
    up1, up1_t = UpBlock2D(input=midblock,temb=time_emb,ci=1280,co=1280,tag="UP1",resconcat=upconcat[0])
    up2, up2_t = CrossAttnUpBlock2D(input=up1,temb=time_emb,txtemb=text_emb,ci=1280,co=1280,tag="UP2",resconcat=upconcat[1])
    up3, up3_t = CrossAttnUpBlock2D(input=up2,temb=time_emb,txtemb=text_emb,ci=1280,co=640,tag="UP3",resconcat=upconcat[2])
    upblock, up4_t = CrossAttnUpBlock2D(input=up3,temb=time_emb,txtemb=text_emb,ci=640,co=320, tag="UP4",isLast=True,resconcat=upconcat[3])

    time = time+db1_t+db2_t+db3_t+db4_t+mid_t+up1_t+up2_t+up3_t+up4_t

    # 5th step, post-pre block
    # ignore group norm and silu, suppose this part will be combined with upblock output
    conv_out = WTConvKernel(co=4,ci=320)
    output, t = op_conv(upblock,conv_out,"post-proc conv_out","GM")
    time += (t,)

    return time


def parseTime(time:Time, tag:str):
    # summary all op time cost.
    print("--------------------> summar %s perf data <--------------------"%tag)
    total_time = 0
    for op in time:
        total_time += op.op_time

    op_ldst = 0
    op_opbmm = 0
    op_opconv = 0
    for op in time:
        op.update(total_time)
        print("%s, %s %s %s to %s ldst=%.2fus, mac=%.2fus"%(op.tag, op.op1_shape, op.op, 
                                                            op.op2_shape, op.out_shape, op.ldst, op.mac))
        if op.bottleneck == "ldst":
            op_ldst += 1
            if op.op == "conv2D":
                op_opconv += 1
            elif op.op == "bmm":
                op_opbmm += 1

    ops = len(time)
    print("*************************************************************************")
    print("*** run per model, %dcore, %dcluster, %dGB/s BW, %dMB GM, %dx%d img size, %d batch, %s percision"%
          (CORE,CLUSTER,BW,GM_MAX_SIZE/2**20,IMG_H,IMG_W,BATCH,PRECISION))
    print("*** %s: ignore activation func (layernorm, groupnorm, silu,etc), eltwise and reshape"%tag)
    print("*** total %d ops, %d ops are mac bottleneck."%(ops, ops-op_ldst))
    print("*** %d ops are ldst bound(%d are bmm, %d are linear, %d are conv)"%(op_ldst, op_opbmm, op_ldst-op_opbmm-op_opconv, op_opconv))
    print("*** %s time cost is %.2fms"%(tag, total_time/1000.0))
    print("*************************************************************************")

def Attn(input:Feature, ci:int, tag:str):
    tag = tag+Attn.__name__

    # TODO add reshape here.
    sample = Feature(input.n,None,None,None,input.h*input.w,input.c,"GM","nld","reshape_to_attn")
    input.release()

    wqkv = WTLinear(ci=ci,co=ci)
    q, q_t = op_linear(sample,wqkv,tag+"->selfattn_q")
    k, k_t = op_linear(sample,wqkv,tag+"->selfattn_k")
    # TODO, transpose k to kt
    k.len = k.dims
    k.dims = q.len
    qkt, qkt_t = op_bmm(q,k,tag+"->setfattn_qkt")
    q.release()
    k.release()
    # ignore scaler and softmax
    v, v_t = op_linear(sample,wqkv,tag+"->selfattn_v")
    attn, attn_t = op_bmm(qkt,v,tag+"->selfattn_score")
    qkt.release()
    v.release()
    wt_attnlinear = WTLinear(ci=ci,co=ci)
    selfattn, selfattn_t = op_linear(attn, wt_attnlinear, tag+"->selfattn_o")
    attn.release()
    # 0.2 TODO residual add
    sample.release()
    t = (q_t,k_t,v_t,qkt_t,attn_t,selfattn_t)

    return selfattn, t

def UpDecoderBlock2D(input:Feature, ci:int, co:int, tag:str):
    resb1, res1_t = ResnetBlock2D(None,input,ci,co,tag+"->layer1",False)
    resb2, res2_t = ResnetBlock2D(None,resb1,co,co,tag+"->layer2",False)
    resb3, res3_t = ResnetBlock2D(None,resb2,co,co,tag+"->layer3",False)
    t = res1_t + res2_t + res3_t
    if tag != "UB4":
        output, up_t = Upsample2D(resb3,co,co,tag+"->Upsample2D")
        t += up_t
        return output, t
    else:
        return resb3, t

def UNetMidBlock2D(input:Feature, ci:int, tag:str):
    resb1, res1_t = ResnetBlock2D(None,input,ci,ci,tag+"->layer1",False)
    attn, attn_t = Attn(resb1,ci,tag)
    # TODO add reshape time here
    res2_in = Feature(n=resb1.n,c=resb1.c,h=resb1.h,w=resb1.w,len=resb1.len,dims=resb1.dims,name=tag+"attn_reshape")
    attn.release()
    resb2, res2_t = ResnetBlock2D(None,res2_in,ci,ci,tag+"->layer2",False)
    t = res1_t + attn_t + res2_t
    return resb2, t

def VAEDecoder():
    # 0 step post_quant and conv in
    time = ()
    latent = Feature(n=BATCH,c=4,h=LATENT_H,w=LATNET_W,format="nchw",location="DDR",name="latent")
    kernel_pq_conv = WTConvKernel(4,4,1,1)
    pqconv_out, pq_conv_t = op_conv(latent,kernel_pq_conv,"post_quant_conv")
    kernel_conv_in = WTConvKernel(512,4)
    conv_in, conv_in_t = op_conv(pqconv_out,kernel_conv_in,"conv_in")
    pqconv_out.release()
    time = (pq_conv_t, conv_in_t)

    # 1 mid block
    mid, mid_t = UNetMidBlock2D(conv_in,512,"Mid")

    # 2 up block
    up1, up1_t = UpDecoderBlock2D(mid,512,512,"UB1")
    up2, up2_t = UpDecoderBlock2D(up1,512,512,"UB2")
    up3, up3_t = UpDecoderBlock2D(up2,512,256,"UB3")
    up_block, up4_t = UpDecoderBlock2D(up3,256,128,"UB4")

    time = time + mid_t + up1_t + up2_t + up3_t + up4_t

    # 3 post conv
    # ignore GroupNorm and Silu
    kernel_conv_out = WTConvKernel(3,128)
    image, conv_out_t = op_conv(up_block,kernel_conv_out,"conv_out","DDR")
    time = time + (conv_out_t,)

    return time

time_unet = Unet()
parseTime(time_unet,"Unet")
# time_vae_decoder = VAEDecoder()
# parseTime(time_vae_decoder,"VAE Decoder")

