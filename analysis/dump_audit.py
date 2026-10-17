"""Audit helper: dump ground-truth aggregates from results/metrics_summary.json
to cross-check the manuscript prose. Structure: d[model][seed][evalset][field].
Run on the GPU host:  python analysis/dump_audit.py
"""
import json
import statistics as st

d = json.load(open("results/metrics_summary.json"))
ES = ["within", "beilunhe", "zhanjiang"]
models = ["unet", "unet_cbam", "deeplabv3plus", "temporal_unet", "unet_cbam_msa", "unet_msa"]
main5 = ["unet", "unet_cbam", "deeplabv3plus", "temporal_unet", "unet_cbam_msa"]


def ms(x):
    return (round(st.mean(x), 3), round(st.stdev(x), 3) if len(x) > 1 else 0.0)


def cm_pred_true(cm):
    tot = sum(sum(r) for r in cm)
    predc = cm[0][1] + cm[0][2] + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] + cm[3][1] + cm[3][2]
    truec = cm[1][0] + cm[1][1] + cm[1][2] + cm[1][3] + cm[2][0] + cm[2][1] + cm[2][2] + cm[2][3]
    return 100 * predc / tot, 100 * truec / tot


agg = {}
print("=== per model x evalset : mean(+/-std over 3 seeds) ===")
for m in models:
    for es in ES:
        rs = [d[m][s][es] for s in d[m]]
        OA = [r["OA"] for r in rs]
        kap = [r["kappa"] for r in rs]
        mIoU = [r["mIoU"] for r in rs]
        F1c = [r["F1_change"] for r in rs]
        IoUc = [r["IoU_change"] for r in rs]
        prec = [r["precision_change"] for r in rs]
        om = [r["Omission"] for r in rs]
        commS = [r["Commission"] for r in rs]
        pred = [cm_pred_true(r["cm"])[0] for r in rs]
        true = [cm_pred_true(r["cm"])[1] for r in rs]
        agg[(m, es)] = dict(OA=ms(OA), kap=ms(kap), mIoU=ms(mIoU), F1c=ms(F1c),
                            IoUc=ms(IoUc), prec=ms(prec), om=ms(om),
                            commS=ms(commS), pred=ms(pred), true=ms(true))
        print("%-16s %-9s OA=%s kap=%s mIoU=%s F1c=%s IoUc=%s prec=%s om=%s commSum=%s pred%%=%s true%%=%s"
              % (m, es, ms(OA), ms(kap), ms(mIoU), ms(F1c), ms(IoUc), ms(prec), ms(om), ms(commS), ms(pred), ms(true)))


def rng(field, eslist=None):
    eslist = ES if eslist is None else eslist
    vals = [agg[(m, es)][field][0] for m in main5 for es in eslist]
    return round(min(vals), 3), round(max(vals), 3)


print("\n=== ranges across 5 main DNNs (3-seed means) ===")
print("OA        all-es=%s indep=%s" % (rng("OA"), rng("OA", ["beilunhe", "zhanjiang"])))
print("mIoU      all-es=%s indep=%s" % (rng("mIoU"), rng("mIoU", ["beilunhe", "zhanjiang"])))
print("F1_change all-es=%s indep=%s" % (rng("F1c"), rng("F1c", ["beilunhe", "zhanjiang"])))
print("IoU_change all-es=%s indep=%s" % (rng("IoUc"), rng("IoUc", ["beilunhe", "zhanjiang"])))
def comm_alt(eslist):
    eslist = ES if eslist is None else eslist
    precs = [agg[(m, es)]["prec"][0] for m in main5 for es in eslist]
    return round(1 - max(precs), 3), round(1 - min(precs), 3)


print("comm=1-prec  all-es=%s indep=%s" % (comm_alt(None), comm_alt(["beilunhe", "zhanjiang"])))
print("Omission  all-es=%s indep=%s" % (rng("om"), rng("om", ["beilunhe", "zhanjiang"])))
print("pred%%     all-es=%s indep=%s" % (rng("pred"), rng("pred", ["beilunhe", "zhanjiang"])))
tlo = round(min(agg[(main5[0], e)]["true"][0] for e in ES), 3)
thi = round(max(agg[(main5[0], e)]["true"][0] for e in ES), 3)
print("true%%     regions min=%.3f max=%.3f  (per-region: within=%.3f beilunhe=%.3f zhanjiang=%.3f)"
      % (tlo, thi, agg[(main5[0], "within")]["true"][0],
         agg[(main5[0], "beilunhe")]["true"][0], agg[(main5[0], "zhanjiang")]["true"][0]))

print("\n=== mIoU deltas per model (mean) ===")
for m in main5:
    wb = round(agg[(m, "beilunhe")]["mIoU"][0] - agg[(m, "within")]["mIoU"][0], 3)
    wz = round(agg[(m, "zhanjiang")]["mIoU"][0] - agg[(m, "within")]["mIoU"][0], 3)
    print("%-16s within->beilunhe %+.3f   within->zhanjiang %+.3f" % (m, wb, wz))

print("\n=== per-class F1 (3-seed mean) ranges across 5 main DNNs x 3 regions ===")
for cls in ["stable_mangrove", "loss", "gain", "stable_background"]:
    vals = [d[m][s][es]["per_class"][cls]["F1"] for m in main5 for s in d[m] for es in ES]
    print("%-16s min=%.3f max=%.3f" % (cls, min(vals), max(vals)))

print("\n=== proposed-model (unet_cbam_msa) key cells ===")
for es in ES:
    a = agg[("unet_cbam_msa", es)]
    print("unet_cbam_msa %-10s OA=%s kap=%s mIoU=%s F1c=%s" % (es, a["OA"], a["kap"], a["mIoU"], a["F1c"]))
