## Object detection module usage:

(ipython) %run animal/object_detection_module/object_collect_data.py --target_dir '/target/data/directory' to collect a dataset.

(ipython) %run animal/object_detection_module/object_train.py --data_dir '/origin/data/directory' --logs_dir '/target/logs/directory' to train a object network.


## Trained network

A trained network checkpoint can be found in ckpt_path="/workspace7/Unity3D/albert/object_module_logs/model_XXX.ckpt".

which can be loaded with the following code snippet:

```python

from animal.object_detection_module import ImpalaCNNObject

object_model= ImpalaCNNObject.load(ckpt_path)

```



#### The network expects:

 - Observations inputs with shape (batch_size, 3, 84, 84)



#### The network returns 4 things:

 - logits of detected objects
 - probabilities of detected object in a vector of length 15. Where:

        -   GoodGoal corresponds to position 0
        -   BadGoal corresponds to position 1
        -   GoodGoalMulti corresponds to position 2
        -   Wall corresponds to position 3
        -   Ramp corresponds to position 4
        -   CylinderTunnel corresponds to position 5
        -   WallTransparent corresponds to position 6
        -   CylinderTunnelTransparent corresponds to position 7
        -   Cardbox1 corresponds to position 8
        -   Cardbox2 corresponds to position 9
        -   UObject corresponds to position 10
        -   LObject corresponds to position 11
        -   LObject2 corresponds to position 12
        -   DeathZone corresponds to position 13
        -   HotZone corresponds to position 14

 - hidden state if trained with recurrence (not the case)
 - deep features before the class prediction (output of the cnn).