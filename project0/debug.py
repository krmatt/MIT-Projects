#pragma: coderesponse template
def get_sum_metrics(predictions, metrics=None):
    if metrics is None:
        metrics = []
        mpop = False
    else:
        mpop = True
    
    for i in range(3):
        metrics.append(lambda x: x + i)

    if mpop:
        metrics.append(metrics.pop(0))

    sum_metrics = 0
    for i in range(len(metrics)):
        sum_metrics += metrics[i](predictions)
        
    return sum_metrics
#pragma: coderesponse end


def main():
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9
    print(get_sum_metrics(3, [lambda x: x]))  # Should be (3) + (3 + 0) + (3 + 1) + (3 + 2) = 15
    print(get_sum_metrics(0))  # Should be (0 + 0) + (0 + 1) + (0 + 2) = 3
    print(get_sum_metrics(1))  # Should be (1 + 0) + (1 + 1) + (1 + 2) = 6
    print(get_sum_metrics(2))  # Should be (2 + 0) + (2 + 1) + (2 + 2) = 9

if __name__ == "__main__":
    main()
