Results so far:

---------12.8 LSTM + ANN------------

**3 CLASS**
{
  "label": "3class",
  "test_loss": 0.36362586199061275,
  "test_acc": 0.6653974847121265,
  "test_f1_macro": 0.551802422699829,
  "class_report": {
    "0": {
      "precision": 0.7927643133122585,
      "recall": 0.8150957024196461,
      "f1-score": 0.8037749287749287,
      "support": 5538.0
    },
    "1": {
      "precision": 0.3333333333333333,
      "recall": 0.3439273552780931,
      "f1-score": 0.33854748603351953,
      "support": 1762.0
    },
    "2": {
      "precision": 0.5601731601731602,
      "recall": 0.47329919531821507,
      "f1-score": 0.5130848532910388,
      "support": 1367.0
    },
    "accuracy": 0.6653974847121265,
    "macro avg": {
      "precision": 0.562090268939584,
      "recall": 0.5441074176719848,
      "f1-score": 0.551802422699829,
      "support": 8667.0
    },
    "weighted avg": {
      "precision": 0.6626766828675816,
      "recall": 0.6653974847121265,
      "f1-score": 0.6633452429209031,
      "support": 8667.0
    }
  },
  "confusion_matrix": [
    [
      4514,
      814,
      210
    ],
    [
      858,
      606,
      298
    ],
    [
      322,
      398,
      647
    ]
  ],
  "best_epoch": 2,
  "elapsed_sec": 433.9,
  "class_counts_train": [
    25801,
    8264,
    6378
  ]
}

**BINARY**
{
  "label": "binary",
  "test_loss": 1.8812104903855262,
  "test_acc": 0.8550824968270451,
  "test_f1_macro": 0.7234904459530638,
  "class_report": {
    "0": {
      "precision": 0.9113803430438334,
      "recall": 0.9171232876712329,
      "f1-score": 0.9142427966680322,
      "support": 7300.0
    },
    "1": {
      "precision": 0.5420136260408781,
      "recall": 0.5237746891002194,
      "f1-score": 0.5327380952380952,
      "support": 1367.0
    },
    "accuracy": 0.8550824968270451,
    "macro avg": {
      "precision": 0.7266969845423558,
      "recall": 0.7204489883857261,
      "f1-score": 0.7234904459530638,
      "support": 8667.0
    },
    "weighted avg": {
      "precision": 0.8531220873448556,
      "recall": 0.8550824968270451,
      "f1-score": 0.854070080981552,
      "support": 8667.0
    }
  },
  "confusion_matrix": [
    [
      6695,
      605
    ],
    [
      651,
      716
    ]
  ],
  "best_epoch": 2,
  "elapsed_sec": 594.5,
  "class_counts_train": [
    34065,
    6378
  ]
}

_________________________________________________


---------13.2 LSTM + ANN + MAMBA------------

**3 CLASS**
{
  "label": "3class",
  "best_epoch": 2,
  "val_best_loss": 0.11032767010435286,
  "test_loss": 0.11282063157545787,
  "test_acc": 0.6843198338525441,
  "test_f1_macro": 0.5783488436922818,
  "confusion_matrix": [
    [
      4538,
      905,
      95
    ],
    [
      757,
      859,
      146
    ],
    [
      229,
      604,
      534
    ]
  ],
  "class_report": {
    "0": {
      "precision": 0.8215061549601738,
      "recall": 0.8194293968941856,
      "f1-score": 0.8204664617609836,
      "support": 5538.0
    },
    "1": {
      "precision": 0.3627533783783784,
      "recall": 0.4875141884222474,
      "f1-score": 0.4159806295399516,
      "support": 1762.0
    },
    "2": {
      "precision": 0.6890322580645162,
      "recall": 0.39063643013899046,
      "f1-score": 0.49859943977591037,
      "support": 1367.0
    },
    "accuracy": 0.6843198338525441,
    "macro avg": {
      "precision": 0.6244305971343561,
      "recall": 0.5658600051518078,
      "f1-score": 0.5783488436922818,
      "support": 8667.0
    },
    "weighted avg": {
      "precision": 0.7073473676758208,
      "recall": 0.6843198338525441,
      "f1-score": 0.6874681629924301,
      "support": 8667.0
    }
  },
  "elapsed_sec": 1059.0,
  "train_shape_dyn": [
    40443,
    24,
    376
  ],
  "train_shape_static": [
    40443,
    70
  ]
}


**BINARY**
{
  "label": "binary",
  "best_epoch": 2,
  "val_best_loss": 0.4428143391165149,
  "test_loss": 0.45727642834207183,
  "test_acc": 0.7819314641744548,
  "test_f1_macro": 0.6986120126644323,
  "confusion_matrix": [
    [
      5667,
      1633
    ],
    [
      257,
      1110
    ]
  ],
  "class_report": {
    "0": {
      "precision": 0.9566171505739365,
      "recall": 0.7763013698630137,
      "f1-score": 0.8570780399274047,
      "support": 7300.0
    },
    "1": {
      "precision": 0.4046664236237696,
      "recall": 0.8119970738844184,
      "f1-score": 0.5401459854014599,
      "support": 1367.0
    },
    "accuracy": 0.7819314641744548,
    "macro avg": {
      "precision": 0.680641787098853,
      "recall": 0.7941492218737161,
      "f1-score": 0.6986120126644323,
      "support": 8667.0
    },
    "weighted avg": {
      "precision": 0.8695608861524667,
      "recall": 0.7819314641744548,
      "f1-score": 0.8070900257890677,
      "support": 8667.0
    }
  },
  "elapsed_sec": 966.9,
  "train_shape_dyn": [
    40443,
    24,
    376
  ],
  "train_shape_static": [
    40443,
    70
  ]
}