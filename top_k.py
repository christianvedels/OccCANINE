from histocc import OccCANINE


EXAMPLES = [
    ["tailor of the finest suits"],
    ["the train's fireman"],
    ["nurse at the local hospital"],
    ["policeman and fisher"],
]

def main():
    # Load model
    model = OccCANINE(batch_size=2, verbose=False)

    # Normal predict method as check
    print(model.predict(EXAMPLES))

    # Top-k predictions
    out = model.predict(
        EXAMPLES,
        prediction_type='greedy-topk',
        order_invariant_conf=False,
    )

    print(out.sort_values('occ1'))


if __name__ == '__main__':
    main()
