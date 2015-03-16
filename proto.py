import caffe.proto.caffe_pb2 as caffe_pb2

import google.protobuf as protobuf

import collections
import sys


class ProtoObject(object):

    def __init__(self, msg_name, **kwargs):
        self._proto = getattr(caffe_pb2, msg_name)
        self._field_info = self._proto.DESCRIPTOR.fields_by_name
        for field, info in self._field_info.items():
            if info.default_value is not None:
                setattr(self, field, info.default_value)
        for key, value in kwargs.items():
            if isinstance(value, dict):
                message_desc = self._field_info[key].message_type
                cls = self._message_type_to_protoclass(message_desc)
                value = cls(message_desc.name, **value)
            setattr(self, key, value)

    def to_message(self):
        message = self._proto()
        for field, info in self._field_info.items():
            if not hasattr(self, field):
                if info.label == info.LABEL_REQUIRED:
                    raise AttributeError('Net missing required field: {}'.format(field))
                continue
            val = getattr(self, field)
            if info.label == info.LABEL_REPEATED:
                if isinstance(val, basestring):
                    raise AttributeError('Net field {} must be sequence, not string'.format(field))
                if info.message_type:
                    sub_msgs = [sub.to_message() for sub in val]
                    getattr(message, field).extend(sub_msgs)
                else:
                    getattr(message, field).extend(val)
            else:
                if info.message_type:
                    getattr(message, field).CopyFrom(val.to_message())
                else:
                    setattr(message, field, val)
        return message

    @staticmethod
    def _message_type_to_protoclass(message_type):
        if message_type.name == 'NetParameter':
            return ProtoNet
        elif message_type.name == 'LayerParameter':
            return ProtoLayer
        else:
            return ProtoParam

    @classmethod
    def from_message(cls, message):
        if message.DESCRIPTOR.name in ('NetParameter', 'LayerParameter'):
            protoobj = cls()
        else:
            protoobj = cls(message.DESCRIPTOR.name)
        protoobj._message = message
        msg_fields = [field[0].name for field in message.ListFields()]
        for field, info in protoobj._field_info.items():
            if field not in msg_fields:
                continue
            val = getattr(message, field)
            if info.message_type:
                if info.label == info.LABEL_REPEATED:
                    protocls = cls._message_type_to_protoclass(info.message_type)
                    setattr(protoobj, field, [protocls.from_message(param_msg) for param_msg in val])
                else:
                    setattr(protoobj, field, ProtoParam.from_message(val))
            else:
                setattr(protoobj, field, val)
        return protoobj

    def __setattr__(self, name, value):
        if name.startswith('_') or name in self._field_info:
            object.__setattr__(self, name, value)
        else:
            raise AttributeError('Invalid field {}'.format(name))


class ProtoNet(ProtoObject):

    def __init__(self, **kwargs):
        ProtoObject.__init__(self, 'NetParameter', **kwargs)

    @staticmethod
    def from_prototxt(inpath):
        message = caffe_pb2.NetParameter()
        with open(inpath, 'r') as f:
            protobuf.text_format.Merge(f.read(), message)
        return ProtoNet.from_message(message)

    def rename_blob(self, old, new):
        for layer in self.layer:
            if hasattr(layer, 'top'):
                for i, name in enumerate(layer.top):
                    if name == old:
                        layer.top[i] = new
            if hasattr(layer, 'bottom'):
                for i, name in enumerate(layer.bottom):
                    if name == old:
                        layer.bottom[i] = new

    def __getitem__(self, name):
        layers = []
        for layer in self.layer:
            if layer.name == name:
                layers.append(layer)
        if not layers:
            return None
        elif len(layers) == 1:
            return layers[0]
        else:
            result = {}
            for layer in layers:
                result[layer.include[0].phase] = layer
            return result

    def write_prototxt(self, outpath):
        with open(outpath, 'w') as f:
            f.write(protobuf.text_format.MessageToString(self.to_message()))


class MetaProtoLayer(type):

    def __getattr__(self, name):
        return FakeLayerClass(name)


class ProtoLayer(ProtoObject):
    __metaclass__ = MetaProtoLayer

    def __init__(self, **kwargs):
        ProtoObject.__init__(self, 'LayerParameter', **kwargs)

    def __repr__(self):
        return '<ProtoLayer {} {}>'.format(self.type, self.name)


class FakeLayerClass(object):

    def __init__(self, name):
        self._proto = getattr(caffe_pb2, 'LayerParameter')
        self.name = name

    def __call__(self, *args, **kwargs):
        return ProtoLayer(type=self.name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._proto, name)


class MetaProtoParam(type):

    def __getattr__(self, name):
        return FakeParamClass(name)


class FakeParamClass(object):

    def __init__(self, name):
        self._proto = getattr(caffe_pb2, name)
        self.name = name

    def __call__(self, *args, **kwargs):
        return ProtoParam(self.name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._proto, name)


class ProtoParam(ProtoObject):
    __metaclass__ = MetaProtoParam

    def __init__(self, msg_name, **kwargs):
        ProtoObject.__init__(self, msg_name, **kwargs)

    def __repr__(self):
        return '<ProtoParam {}>'.format(self._proto.DESCRIPTOR.name)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        arg = sys.argv[1]
    else:
        arg = '/home/tzeng/src/caffe/models/bvlc_reference_caffenet/train_val.prototxt'
    net = ProtoNet.from_prototxt(arg)
    msg = net.to_message()
